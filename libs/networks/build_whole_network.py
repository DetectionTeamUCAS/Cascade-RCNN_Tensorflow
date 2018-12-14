# -*-coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from libs.networks import resnet
from libs.networks import mobilenet_v2
from libs.box_utils import encode_and_decode
from libs.box_utils import boxes_utils
from libs.box_utils import anchor_utils
from libs.configs import cfgs
from libs.losses import losses
from libs.losses import tfapi_loss
from libs.box_utils import show_box_in_tensor
from libs.detection_oprations.proposal_opr import postprocess_rpn_proposals
from libs.detection_oprations.anchor_target_layer_without_boxweight import anchor_target_layer
from libs.detection_oprations.proposal_target_layer import proposal_target_layer
from tensorflow.python import pywrap_tensorflow

class DetectionNetwork(object):

    def __init__(self, base_network_name, is_training):

        self.base_network_name = base_network_name
        self.is_training = is_training
        self.num_anchors_per_location = len(cfgs.ANCHOR_SCALES) * len(cfgs.ANCHOR_RATIOS)

    def build_base_network(self, input_img_batch):

        if self.base_network_name.startswith('resnet_v1'):
            return resnet.resnet_base(input_img_batch, scope_name=self.base_network_name, is_training=self.is_training)

        elif self.base_network_name.startswith('MobilenetV2'):
            return mobilenet_v2.mobilenetv2_base(input_img_batch, is_training=self.is_training)

        else:
            raise ValueError('Sry, we only support resnet or mobilenet_v2')

    def postprocess_fastrcnn(self, rois, bbox_ppred, scores, img_shape, scope):
        '''

        :param rois:[-1, 4]
        :param bbox_ppred: [-1, (cfgs.Class_num+1) * 4]
        :param scores: [-1, cfgs.Class_num + 1]
        :return:
        '''

        with tf.name_scope('postprocess_fastrcnn_{}'.format(scope)):
            rois = tf.stop_gradient(rois)
            scores = tf.stop_gradient(scores)
            bbox_ppred = tf.reshape(bbox_ppred, [-1, cfgs.CLASS_NUM + 1, 4])
            bbox_ppred = tf.stop_gradient(bbox_ppred)

            bbox_pred_list = tf.unstack(bbox_ppred, axis=1)
            score_list = tf.unstack(scores, axis=1)

            allclasses_boxes = []
            allclasses_scores = []
            categories = []
            for i in range(1, cfgs.CLASS_NUM+1):

                # 1. decode boxes in each class
                tmp_encoded_box = bbox_pred_list[i]
                tmp_score = score_list[i]
                tmp_decoded_boxes = encode_and_decode.decode_boxes(encoded_boxes=tmp_encoded_box,
                                                                   reference_boxes=rois,
                                                                   scale_factors=cfgs.ROI_SCALE_FACTORS)
                # tmp_decoded_boxes = encode_and_decode.decode_boxes(boxes=rois,
                #                                                    deltas=tmp_encoded_box,
                #                                                    scale_factor=cfgs.ROI_SCALE_FACTORS)

                # 2. clip to img boundaries
                tmp_decoded_boxes = boxes_utils.clip_boxes_to_img_boundaries(decode_boxes=tmp_decoded_boxes,
                                                                             img_shape=img_shape)

                # 3. NMS
                keep = tf.image.non_max_suppression(
                    boxes=tmp_decoded_boxes,
                    scores=tmp_score,
                    max_output_size=cfgs.FAST_RCNN_NMS_MAX_BOXES_PER_CLASS,
                    iou_threshold=cfgs.FAST_RCNN_NMS_IOU_THRESHOLD)

                perclass_boxes = tf.gather(tmp_decoded_boxes, keep)
                perclass_scores = tf.gather(tmp_score, keep)

                allclasses_boxes.append(perclass_boxes)
                allclasses_scores.append(perclass_scores)
                categories.append(tf.ones_like(perclass_scores) * i)

            final_boxes = tf.concat(allclasses_boxes, axis=0)
            final_scores = tf.concat(allclasses_scores, axis=0)
            final_category = tf.concat(categories, axis=0)

            if self.is_training:
                '''
                in training. We should show the detecitons in the tensorboard. So we add this.
                '''
                kept_indices = tf.reshape(tf.where(tf.greater_equal(final_scores, cfgs.SHOW_SCORE_THRSHOLD)), [-1])

                final_boxes = tf.gather(final_boxes, kept_indices)
                final_scores = tf.gather(final_scores, kept_indices)
                final_category = tf.gather(final_category, kept_indices)

        return final_boxes, final_scores, final_category

    def postprocess_cascade(self, rois, bbox_ppred, scope):
        '''

        :param rois:[-1, 4]
        :param bbox_ppred: bbox_ppred: [-1, 4]
        :param scores: [-1, 1]
        :return:

        '''

        with tf.name_scope('postprocess_cascade_{}'.format(scope)):
            rois = tf.stop_gradient(rois)
            bbox_ppred = tf.stop_gradient(bbox_ppred)

            # decode boxes
            decoded_boxes = encode_and_decode.decode_boxes(encoded_boxes=bbox_ppred,
                                                           reference_boxes=rois,
                                                           scale_factors=cfgs.ROI_SCALE_FACTORS)

        return decoded_boxes

    def roi_pooling(self, feature_maps, rois, img_shape):
        '''
        Here use roi warping as roi_pooling

        :param featuremaps_dict: feature map to crop
        :param rois: shape is [-1, 4]. [x1, y1, x2, y2]
        :return:
        '''

        with tf.variable_scope('ROI_Warping'):
            img_h, img_w = tf.cast(img_shape[1], tf.float32), tf.cast(img_shape[2], tf.float32)
            N = tf.shape(rois)[0]
            x1, y1, x2, y2 = tf.unstack(rois, axis=1)

            normalized_x1 = x1 / img_w
            normalized_x2 = x2 / img_w
            normalized_y1 = y1 / img_h
            normalized_y2 = y2 / img_h

            normalized_rois = tf.transpose(
                tf.stack([normalized_y1, normalized_x1, normalized_y2, normalized_x2]), name='get_normalized_rois')

            normalized_rois = tf.stop_gradient(normalized_rois)

            cropped_roi_features = tf.image.crop_and_resize(feature_maps, normalized_rois,
                                                            box_ind=tf.zeros(shape=[N, ],
                                                                             dtype=tf.int32),
                                                            crop_size=[cfgs.ROI_SIZE, cfgs.ROI_SIZE],
                                                            name='CROP_AND_RESIZE'
                                                            )
            roi_features = slim.max_pool2d(cropped_roi_features,
                                           [cfgs.ROI_POOL_KERNEL_SIZE, cfgs.ROI_POOL_KERNEL_SIZE],
                                           stride=cfgs.ROI_POOL_KERNEL_SIZE)

        return roi_features


    def build_fastrcnn(self, feature_to_cropped, rois, img_shape, scope):

        with tf.variable_scope('Fast-RCNN_{}'.format(scope)):
            # 5. ROI Pooling
            with tf.variable_scope('rois_pooling'):
                pooled_features = self.roi_pooling(feature_maps=feature_to_cropped, rois=rois, img_shape=img_shape)

            # 6. inferecne rois in Fast-RCNN to obtain fc_flatten features
            if self.base_network_name.startswith('resnet'):
                fc_flatten = resnet.restnet_head(input=pooled_features,
                                                 is_training=self.is_training,
                                                 scope_name=self.base_network_name,
                                                 stage=scope)
            elif self.base_network_name.startswith('Mobile'):
                fc_flatten = mobilenet_v2.mobilenetv2_head(inputs=pooled_features,
                                                           is_training=self.is_training,
                                                           stage=scope)
            else:
                raise NotImplementedError('only support resnet and mobilenet')

            # 7. cls and reg in Fast-RCNN
            # tf.variance_scaling_initializer()
            # tf.VarianceScaling()
            with slim.arg_scope([slim.fully_connected], weights_regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY)):

                cls_score = slim.fully_connected(fc_flatten,
                                                 num_outputs=cfgs.CLASS_NUM+1,
                                                 weights_initializer=slim.variance_scaling_initializer(factor=1.0,
                                                                                                       mode='FAN_AVG',
                                                                                                       uniform=True),
                                                 activation_fn=None, trainable=self.is_training,
                                                 scope='cls_fc')

                bbox_pred = slim.fully_connected(fc_flatten,
                                                 num_outputs=(cfgs.CLASS_NUM+1)*4,
                                                 weights_initializer=slim.variance_scaling_initializer(factor=1.0,
                                                                                                       mode='FAN_AVG',
                                                                                                       uniform=True),
                                                activation_fn=None, trainable=self.is_training,
                                                 scope='reg_fc')
                if not scope == 'stage3':
                    # for convient. It also produce (cls_num +1) bboxes
                    cls_score = tf.reshape(cls_score, [-1, cfgs.CLASS_NUM+1])
                    bbox_pred = tf.reshape(bbox_pred, [-1, 4*(cfgs.CLASS_NUM+1)])
                    bbox_pred_ins = tf.reshape(bbox_pred, [-1, cfgs.CLASS_NUM+1, 4])

                    # only keep a box which score is the bigest
                    keep_abox = tf.argmax(cls_score, axis=1)
                    keep_inds = tf.reshape(tf.transpose(tf.stack([tf.cumsum(tf.ones_like(keep_abox))-1, keep_abox])), [-1, 2])
                    bbox_pred_fliter = tf.reshape(tf.gather_nd(bbox_pred_ins, keep_inds), [-1, 4])

                    return bbox_pred_fliter, bbox_pred, cls_score
                else:
                    cls_score = tf.reshape(cls_score, [-1, cfgs.CLASS_NUM + 1])
                    bbox_pred = tf.reshape(bbox_pred, [-1, 4 * (cfgs.CLASS_NUM + 1)])
                    return bbox_pred, cls_score

    def add_anchor_img_smry(self, img, anchors, labels):

        positive_anchor_indices = tf.reshape(tf.where(tf.greater_equal(labels, 1)), [-1])
        negative_anchor_indices = tf.reshape(tf.where(tf.equal(labels, 0)), [-1])

        positive_anchor = tf.gather(anchors, positive_anchor_indices)
        negative_anchor = tf.gather(anchors, negative_anchor_indices)

        pos_in_img = show_box_in_tensor.only_draw_boxes(img_batch=img,
                                                        boxes=positive_anchor)
        neg_in_img = show_box_in_tensor.only_draw_boxes(img_batch=img,
                                                        boxes=negative_anchor)

        tf.summary.image('positive_anchor', pos_in_img)
        tf.summary.image('negative_anchors', neg_in_img)

    def add_roi_batch_img_smry(self, img, rois, labels, scope):
        positive_roi_indices = tf.reshape(tf.where(tf.greater_equal(labels, 1)), [-1])

        negative_roi_indices = tf.reshape(tf.where(tf.equal(labels, 0)), [-1])

        pos_roi = tf.gather(rois, positive_roi_indices)
        neg_roi = tf.gather(rois, negative_roi_indices)


        pos_in_img = show_box_in_tensor.only_draw_boxes(img_batch=img,
                                                               boxes=pos_roi)
        neg_in_img = show_box_in_tensor.only_draw_boxes(img_batch=img,
                                                               boxes=neg_roi)
        tf.summary.image('pos_rois_{}'.format(scope), pos_in_img)
        tf.summary.image('neg_rois_{}'.format(scope), neg_in_img)

    def build_loss(self, rpn_box_pred, rpn_bbox_targets, rpn_cls_score, rpn_labels,
                   bbox_pred, bbox_targets, stage2_bbox_pred, stage2_bbox_targets,  stage1_bbox_pred, stage1_bbox_targets,
                   cls_score, labels, stage2_cls_score, stage2_labels, stage1_cls_score, stage1_labels):
        '''

        :param rpn_box_pred: [-1, 4]
        :param rpn_bbox_targets: [-1, 4]
        :param rpn_cls_score: [-1]
        :param rpn_labels: [-1]
        :param bbox_pred: [-1, 4*(cls_num+1)]
        :param bbox_targets: [-1, 4*(cls_num+1)]
        :param cls_score: [-1, cls_num+1]
        :param labels: [-1]
        :return:
        '''
        with tf.variable_scope('build_loss') as sc:
            with tf.variable_scope('rpn_loss'):

                rpn_bbox_loss = losses.smooth_l1_loss_rpn(bbox_pred=rpn_box_pred,
                                                          bbox_targets=rpn_bbox_targets,
                                                          label=rpn_labels,
                                                          sigma=cfgs.RPN_SIGMA)
                # rpn_bbox_loss = tfapi_loss.smooth_l1_loss_rpn(bbox_pred=rpn_box_pred,
                #                                               bbox_targets=rpn_bbox_targets,
                #                                               label=rpn_labels,
                #                                               sigma=cfgs.RPN_SIGMA)
                # rpn_cls_loss:
                # rpn_cls_score = tf.reshape(rpn_cls_score, [-1, 2])
                # rpn_labels = tf.reshape(rpn_labels, [-1])
                # ensure rpn_labels shape is [-1]
                rpn_select = tf.reshape(tf.where(tf.not_equal(rpn_labels, -1)), [-1])
                rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
                rpn_labels = tf.reshape(tf.gather(rpn_labels, rpn_select), [-1])

                rpn_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score,
                                                                                             labels=rpn_labels))

                rpn_cls_loss = rpn_cls_loss * cfgs.RPN_CLASSIFICATION_LOSS_WEIGHT
                rpn_bbox_loss = rpn_bbox_loss * cfgs.RPN_LOCATION_LOSS_WEIGHT

            with tf.variable_scope('FastRCNN_loss'):
                if not cfgs.FAST_RCNN_MINIBATCH_SIZE == -1:
                    bbox_loss = losses.smooth_l1_loss_rcnn(bbox_pred=bbox_pred,
                                                           bbox_targets=bbox_targets,
                                                           label=labels,
                                                           num_classes=cfgs.CLASS_NUM + 1,
                                                           sigma=cfgs.FASTRCNN_SIGMA)
                    stage2_bbox_loss = losses.smooth_l1_loss_rcnn(bbox_pred=stage2_bbox_pred,
                                                           bbox_targets=stage2_bbox_targets,
                                                           label=stage2_labels,
                                                           num_classes=cfgs.CLASS_NUM + 1,
                                                           sigma=cfgs.FASTRCNN_SIGMA)
                    stage1_bbox_loss = losses.smooth_l1_loss_rcnn(bbox_pred=stage1_bbox_pred,
                                                           bbox_targets=stage1_bbox_targets,
                                                           label=stage1_labels,
                                                           num_classes=cfgs.CLASS_NUM + 1,
                                                           sigma=cfgs.FASTRCNN_SIGMA)
                    cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=cls_score, labels=labels))  # beacause already sample before
                    stage2_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=stage2_cls_score, labels=stage2_labels))  # beacause already sample before
                    stage1_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=stage1_cls_score, labels=stage1_labels))  # beacause already sample before

                    bbox_loss = bbox_loss + stage2_bbox_loss + stage1_bbox_loss
                    cls_loss = cls_loss + stage2_cls_loss + stage1_cls_loss
                else:
                    ''' 
                    applying OHEM here
                    '''
                    print(20 * "@@")
                    print("@@" + 10 * " " + "TRAIN WITH OHEM ...")
                    print(20 * "@@")
                    cls_loss, bbox_loss = losses.sum_ohem_loss(cls_score=cls_score,
                                                               label=labels,
                                                               bbox_targets=bbox_targets,
                                                               bbox_pred=bbox_pred,
                                                               num_ohem_samples=256,
                                                               num_classes=cfgs.CLASS_NUM + 1)
                cls_loss = cls_loss * cfgs.FAST_RCNN_CLASSIFICATION_LOSS_WEIGHT
                bbox_loss = bbox_loss * cfgs.FAST_RCNN_LOCATION_LOSS_WEIGHT
            loss_dict = {
                'rpn_cls_loss': rpn_cls_loss,
                'rpn_loc_loss': rpn_bbox_loss,
                'fastrcnn_cls_loss': cls_loss,
                'fastrcnn_loc_loss': bbox_loss
            }
        return loss_dict

    def build_whole_detection_network(self, input_img_batch, gtboxes_batch):

        if self.is_training:
            # ensure shape is [M, 5]
            gtboxes_batch = tf.reshape(gtboxes_batch, [-1, 5])
            gtboxes_batch = tf.cast(gtboxes_batch, tf.float32)

        img_shape = tf.shape(input_img_batch)

        # 1. build base network
        feature_to_cropped = self.build_base_network(input_img_batch)

        # 2. build rpn
        with tf.variable_scope('build_rpn',
                               regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY)):

            rpn_conv3x3 = slim.conv2d(
                feature_to_cropped, 512, [3, 3],
                trainable=self.is_training, weights_initializer=cfgs.INITIALIZER,
                activation_fn=tf.nn.relu,
                scope='rpn_conv/3x3')
            rpn_cls_score = slim.conv2d(rpn_conv3x3, self.num_anchors_per_location*2, [1, 1],stride=1,
                                        trainable=self.is_training, weights_initializer=cfgs.INITIALIZER,
                                        activation_fn=None,
                                        scope='rpn_cls_score')
            rpn_box_pred = slim.conv2d(rpn_conv3x3, self.num_anchors_per_location*4, [1, 1], stride=1,
                                       trainable=self.is_training, weights_initializer=cfgs.BBOX_INITIALIZER,
                                       activation_fn=None,
                                       scope='rpn_bbox_pred')
            rpn_box_pred = tf.reshape(rpn_box_pred, [-1, 4])
            rpn_cls_score = tf.reshape(rpn_cls_score, [-1, 2])
            rpn_cls_prob = slim.softmax(rpn_cls_score, scope='rpn_cls_prob')

        # 3. generate_anchors
        featuremap_height, featuremap_width = tf.shape(feature_to_cropped)[1], tf.shape(feature_to_cropped)[2]
        featuremap_height = tf.cast(featuremap_height, tf.float32)
        featuremap_width = tf.cast(featuremap_width, tf.float32)

        anchors = anchor_utils.make_anchors(base_anchor_size=cfgs.BASE_ANCHOR_SIZE_LIST[0],
                                            anchor_scales=cfgs.ANCHOR_SCALES, anchor_ratios=cfgs.ANCHOR_RATIOS,
                                            featuremap_height=featuremap_height,
                                            featuremap_width=featuremap_width,
                                            stride=cfgs.ANCHOR_STRIDE,
                                            name="make_anchors_forRPN")

        # with tf.variable_scope('make_anchors'):
        #     anchors = anchor_utils.make_anchors(height=featuremap_height,
        #                                         width=featuremap_width,
        #                                         feat_stride=cfgs.ANCHOR_STRIDE[0],
        #                                         anchor_scales=cfgs.ANCHOR_SCALES,
        #                                         anchor_ratios=cfgs.ANCHOR_RATIOS, base_size=16
        #                                         )

        # 4. postprocess rpn proposals. such as: decode, clip, NMS
        with tf.variable_scope('postprocess_RPN'):
            # rpn_cls_prob = tf.reshape(rpn_cls_score, [-1, 2])
            # rpn_cls_prob = slim.softmax(rpn_cls_prob, scope='rpn_cls_prob')
            # rpn_box_pred = tf.reshape(rpn_box_pred, [-1, 4])
            rois, roi_scores = postprocess_rpn_proposals(rpn_bbox_pred=rpn_box_pred,
                                                         rpn_cls_prob=rpn_cls_prob,
                                                         img_shape=img_shape,
                                                         anchors=anchors,
                                                         is_training=self.is_training)
            # rois shape [-1, 4]
            # +++++++++++++++++++++++++++++++++++++add img smry+++++++++++++++++++++++++++++++++++++++++++++++++++++++

            if self.is_training:
                rois_in_img = show_box_in_tensor.draw_boxes_with_scores(img_batch=input_img_batch,
                                                                        boxes=rois,
                                                                        scores=roi_scores)
                tf.summary.image('all_rpn_rois', rois_in_img)

                score_gre_05 = tf.reshape(tf.where(tf.greater_equal(roi_scores, 0.5)), [-1])
                score_gre_05_rois = tf.gather(rois, score_gre_05)
                score_gre_05_score = tf.gather(roi_scores, score_gre_05)
                score_gre_05_in_img = show_box_in_tensor.draw_boxes_with_scores(img_batch=input_img_batch,
                                                                                boxes=score_gre_05_rois,
                                                                                scores=score_gre_05_score)
                tf.summary.image('score_greater_05_rois', score_gre_05_in_img)
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        if self.is_training:
            with tf.variable_scope('sample_anchors_minibatch'):
                rpn_labels, rpn_bbox_targets = \
                    tf.py_func(
                        anchor_target_layer,
                        [gtboxes_batch, img_shape, anchors],
                        [tf.float32, tf.float32])
                rpn_bbox_targets = tf.reshape(rpn_bbox_targets, [-1, 4])
                rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
                rpn_labels = tf.reshape(rpn_labels, [-1])
                self.add_anchor_img_smry(input_img_batch, anchors, rpn_labels)

            # --------------------------------------add smry----------------------------------------------------------------

            rpn_cls_category = tf.argmax(rpn_cls_prob, axis=1)
            kept_rpppn = tf.reshape(tf.where(tf.not_equal(rpn_labels, -1)), [-1])
            rpn_cls_category = tf.gather(rpn_cls_category, kept_rpppn)
            acc = tf.reduce_mean(tf.to_float(tf.equal(rpn_cls_category, tf.to_int64(tf.gather(rpn_labels, kept_rpppn)))))
            tf.summary.scalar('ACC/rpn_accuracy', acc)

            with tf.control_dependencies([rpn_labels]):
                with tf.variable_scope('sample_RCNN_minibatch_stage1'):
                    stage1_rois, stage1_labels, stage1_bbox_targets = \
                    tf.py_func(proposal_target_layer,
                               [rois, gtboxes_batch, cfgs.FAST_RCNN_IOU_POSITIVE_THRESHOLD[0]],
                               [tf.float32, tf.float32, tf.float32])
                    stage1_rois = tf.reshape(stage1_rois, [-1, 4])
                    stage1_labels = tf.to_int32(stage1_labels)
                    stage1_labels = tf.reshape(stage1_labels, [-1])
                    stage1_bbox_targets = tf.reshape(stage1_bbox_targets, [-1, 4*(cfgs.CLASS_NUM+1)])
                    self.add_roi_batch_img_smry(input_img_batch, stage1_rois, stage1_labels, 'stage1')
        else:
            stage1_rois = rois

        # -------------------------------------------------------------------------------------------------------------#
        #                                            Fast-RCNN-before1                                                 #
        # -------------------------------------------------------------------------------------------------------------#

        # 5. build Fast-RCNN-before1
        # rois = tf.Print(rois, [tf.shape(rois)], 'rois shape', summarize=10)
        stage1_bbox_pred_fliter, stage1_bbox_pred, stage1_cls_score = self.build_fastrcnn(feature_to_cropped=feature_to_cropped,
                                                                                          rois=stage1_rois,
                                                                                          img_shape=img_shape,
                                                                                          scope='stage1')
        # bbox_pred shape: [-1, 4*(cls_num+1)].
        # cls_score shape： [-1, cls_num+1]

        stage1_cls_prob = slim.softmax(stage1_cls_score, 'stage1_cls_prob')
        stage1_cls_category = tf.argmax(stage1_cls_prob, axis=1)
        # ----------------------------------------------add smry-------------------------------------------------------
        if self.is_training:
            stage1_fast_acc = tf.reduce_mean(tf.to_float(tf.equal(stage1_cls_category, tf.to_int64(stage1_labels))))
            tf.summary.scalar('ACC/stage1_fast_acc', stage1_fast_acc)

        #  postprocess_fastrcnn_before1
        stage1_bbox = self.postprocess_cascade(rois=stage1_rois,
                                               bbox_ppred=stage1_bbox_pred_fliter,
                                               scope='stage1')

        if self.is_training:
            with tf.control_dependencies([stage1_bbox]):
                with tf.variable_scope('sample_RCNN_minibatch_stage2'):
                    stage2_rois, stage2_labels, stage2_bbox_targets = \
                    tf.py_func(proposal_target_layer,
                               [stage1_bbox, gtboxes_batch, cfgs.FAST_RCNN_IOU_POSITIVE_THRESHOLD[1]],
                               [tf.float32, tf.float32, tf.float32])
                    stage2_rois = tf.reshape(stage2_rois, [-1, 4])
                    stage2_labels = tf.to_int32(stage2_labels)
                    stage2_labels = tf.reshape(stage2_labels, [-1])
                    stage2_bbox_targets = tf.reshape(stage2_bbox_targets, [-1, 4*(cfgs.CLASS_NUM+1)])
                    self.add_roi_batch_img_smry(input_img_batch, stage2_rois, stage2_labels, 'stage2')
        else:
            stage2_rois = stage1_bbox

        # -------------------------------------------------------------------------------------------------------------#
        #                                            Fast-RCNN-before2                                                 #
        # -------------------------------------------------------------------------------------------------------------#

        # 6. build Fast-RCNN-before2
        # rois = tf.Print(rois, [tf.shape(rois)], 'rois shape', summarize=10)
        # stage2_rois = tf.stop_gradient(stage2_rois)
        stage2_bbox_pred_fliter, stage2_bbox_pred, stage2_cls_score = self.build_fastrcnn(feature_to_cropped=feature_to_cropped,
                                                                                          rois=stage2_rois,
                                                                                          img_shape=img_shape,
                                                                                          scope='stage2')
        # bbox_pred shape: [-1, 4*(cls_num+1)].
        # cls_score shape： [-1, cls_num+1]

        stage2_cls_prob = slim.softmax(stage2_cls_score, 'stage2_cls_prob')
        stage2_cls_category = tf.argmax(stage2_cls_prob, axis=1)
        # ----------------------------------------------add smry-------------------------------------------------------
        if self.is_training:
            stage2_fast_acc = tf.reduce_mean(tf.to_float(tf.equal(stage2_cls_category, tf.to_int64(stage2_labels))))
            tf.summary.scalar('ACC/stage2_fast_acc', stage2_fast_acc)

        #  postprocess_fastrcnn_before2
        stage2_bbox = self.postprocess_cascade(rois=stage2_rois,
                                               bbox_ppred=stage2_bbox_pred_fliter,
                                               scope='stage2')

        if self.is_training:
            with tf.control_dependencies([stage2_bbox]):
                with tf.variable_scope('sample_RCNN_minibatch_stage3'):
                    stage3_rois, stage3_labels, stage3_bbox_targets = \
                    tf.py_func(proposal_target_layer,
                               [stage2_rois, gtboxes_batch, cfgs.FAST_RCNN_IOU_POSITIVE_THRESHOLD[2]],
                               [tf.float32, tf.float32, tf.float32])
                    stage3_rois = tf.reshape(stage3_rois, [-1, 4])
                    stage3_labels = tf.to_int32(stage3_labels)
                    stage3_labels = tf.reshape(stage3_labels, [-1])
                    stage3_bbox_targets = tf.reshape(stage3_bbox_targets, [-1, 4*(cfgs.CLASS_NUM+1)])
                    self.add_roi_batch_img_smry(input_img_batch, stage3_rois, stage3_labels, 'stage3')
        else:
            stage3_rois = stage2_bbox

        # -------------------------------------------------------------------------------------------------------------#
        #                                            Fast-RCNN                                                         #
        # -------------------------------------------------------------------------------------------------------------#

        # 7. build Fast-RCNN
        # rois = tf.Print(rois, [tf.shape(rois)], 'rois shape', summarize=10)
        # stage3_rois = tf.stop_gradient(stage3_rois)
        stage3_bbox_pred, stage3_cls_score = self.build_fastrcnn(feature_to_cropped=feature_to_cropped,
                                                                 rois=stage3_rois,
                                                                 img_shape=img_shape,
                                                                 scope='stage3')
        # bbox_pred shape: [-1, 4*(cls_num+1)].
        # cls_score shape： [-1, cls_num+1]

        stage3_cls_prob = slim.softmax(stage3_cls_score, 'stage3_cls_prob')
        stage3_cls_category = tf.argmax(stage3_cls_prob, axis=1)

        # ----------------------------------------------add smry-------------------------------------------------------
        if self.is_training:
            fast_acc = tf.reduce_mean(tf.to_float(tf.equal(stage3_cls_category, tf.to_int64(stage3_labels))))
            tf.summary.scalar('ACC/fast_acc', fast_acc)

        #  postprocess_fastrcnn
        if not self.is_training:
            with slim.arg_scope([slim.conv2d, slim.conv2d_in_plane,
                                 slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
                                reuse=True):
                _, _, final_scores_stage2 = self.build_fastrcnn(feature_to_cropped=feature_to_cropped,
                                                                rois=stage3_rois,
                                                                img_shape=img_shape,
                                                                scope='stage2')
                final_scores_stage2= slim.softmax(final_scores_stage2, 'final_scores_stage2')

                _, _, final_scores_stage1 = self.build_fastrcnn(feature_to_cropped=feature_to_cropped,
                                                                rois=stage3_rois,
                                                                img_shape=img_shape,
                                                                scope='stage1')
                final_scores_stage1 = slim.softmax(final_scores_stage1, 'final_scores_stage1')
                # choose which stage to export
                cls_prob = tf.add(final_scores_stage2, final_scores_stage1)
                cls_prob = tf.add(cls_prob, stage3_cls_prob)/3
                return self.postprocess_fastrcnn(rois=stage3_rois, bbox_ppred=stage3_bbox_pred, scores=cls_prob, img_shape=img_shape, scope='stage3')
        else:
            '''
            when trian. We need build Loss
            '''
            loss_dict = self.build_loss(rpn_box_pred=rpn_box_pred,
                                        rpn_bbox_targets=rpn_bbox_targets,
                                        rpn_cls_score=rpn_cls_score,
                                        rpn_labels=rpn_labels,
                                        bbox_pred=stage3_bbox_pred,
                                        bbox_targets=stage3_bbox_targets,
                                        stage2_bbox_pred=stage2_bbox_pred,
                                        stage2_bbox_targets=stage2_bbox_targets,
                                        stage1_bbox_pred=stage1_bbox_pred,
                                        stage1_bbox_targets=stage1_bbox_targets,
                                        cls_score=stage3_cls_score,
                                        labels=stage3_labels,
                                        stage2_cls_score=stage2_cls_score,
                                        stage2_labels = stage2_labels,
                                        stage1_cls_score=stage1_cls_score,
                                        stage1_labels=stage1_labels)
            final_bbox, final_scores, final_category = self.postprocess_fastrcnn(rois=stage3_rois,
                                                                                 bbox_ppred=stage3_bbox_pred,
                                                                                 scores=stage3_cls_prob,
                                                                                 img_shape=img_shape,
                                                                                 scope='stage3')
            return final_bbox, final_scores, final_category, loss_dict

    def get_restorer(self):
        checkpoint_path = tf.train.latest_checkpoint(os.path.join(cfgs.TRAINED_CKPT, cfgs.VERSION))

        if checkpoint_path != None:
            if cfgs.RESTORE_FROM_RPN:
                print('___restore from rpn___')
                model_variables = slim.get_model_variables()
                restore_variables = [var for var in model_variables if not var.name.startswith('FastRCNN_Head')] + \
                                    [slim.get_or_create_global_step()]
                for var in restore_variables:
                    print(var.name)
                restorer = tf.train.Saver(restore_variables)
            else:
                restorer = tf.train.Saver()
                print("model restore from :", checkpoint_path)

        else:
            checkpoint_path = cfgs.PRETRAINED_CKPT
            print("model restore from pretrained mode, path is :", checkpoint_path)

            model_variables = slim.get_model_variables()
            # for var in model_variables:
            #     print(var.name)
            # print(20*"__++__++__")

            def name_in_ckpt_rpn(var):
                return var.op.name

            def name_in_ckpt_fastrcnn_head(var):
                '''
                Fast-RCNN/resnet_v1_50/block4 -->resnet_v1_50/block4
                Fast-RCNN/MobilenetV2/** -- > MobilenetV2 **
                :param var:
                :return:
                '''
                return '/'.join(var.op.name.split('/')[1:])

            nameInCkpt_Var_dict = {}
            for var in model_variables:
                if var.name.startswith('Fast-RCNN_stage1/'+self.base_network_name):  # +'/block4'
                    var_name_in_ckpt = name_in_ckpt_fastrcnn_head(var)
                    nameInCkpt_Var_dict[var_name_in_ckpt] = var
                else:
                    if var.name.startswith(self.base_network_name):
                        var_name_in_ckpt = name_in_ckpt_rpn(var)
                        nameInCkpt_Var_dict[var_name_in_ckpt] = var
                    else:
                        continue
            restore_variables = nameInCkpt_Var_dict
            for key, item in restore_variables.items():
                print("var_in_graph: ", item.name)
                print("var_in_ckpt: ", key)
                print(20*"___")
            restorer = tf.train.Saver(restore_variables)
            print(20 * "****")
            print("restore from pretrained_weighs in IMAGE_NET")
        return restorer, checkpoint_path

    def get_gradients(self, optimizer, loss):
        '''

        :param optimizer:
        :param loss:
        :return:

        return vars and grads that not be fixed
        '''

        # if cfgs.FIXED_BLOCKS > 0:
        #     trainable_vars = tf.trainable_variables()
        #     # trained_vars = slim.get_trainable_variables()
        #     start_names = [cfgs.NET_NAME + '/block%d'%i for i in range(1, cfgs.FIXED_BLOCKS+1)] + \
        #                   [cfgs.NET_NAME + '/conv1']
        #     start_names = tuple(start_names)
        #     trained_var_list = []
        #     for var in trainable_vars:
        #         if not var.name.startswith(start_names):
        #             trained_var_list.append(var)
        #     # slim.learning.train()
        #     grads = optimizer.compute_gradients(loss, var_list=trained_var_list)
        #     return grads
        # else:
        #     return optimizer.compute_gradients(loss)
        return optimizer.compute_gradients(loss)

    def enlarge_gradients_for_bias(self, gradients):

        final_gradients = []
        with tf.variable_scope("Gradient_Mult") as scope:
            for grad, var in gradients:
                scale = 1.0
                if cfgs.MUTILPY_BIAS_GRADIENT and './biases' in var.name:
                    scale = scale * cfgs.MUTILPY_BIAS_GRADIENT
                if not np.allclose(scale, 1.0):
                    grad = tf.multiply(grad, scale)
                final_gradients.append((grad, var))
        return final_gradients




















