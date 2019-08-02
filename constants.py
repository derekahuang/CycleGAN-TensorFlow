FEATKEY_NR_IMAGES = "nr_images"
FEATKEY_IMAGE_HEIGHT = "height"
FEATKEY_IMAGE_WIDTH = "width"
FEATKEY_ORIGINAL_HEIGHT = "original_height"
FEATKEY_ORIGINAL_WIDTH = "original_width"
FEATKEY_PATIENT_ID = "patient_id"
FEATKEY_LATERALITY = "laterality"
FEATKEY_VIEW = "view"
FEATKEY_IMAGE = "pixel_arr"
FEATKEY_IMAGE_ORIG = "pixel_arr_orig"
FEATKEY_MANUFACTURER = "manufacturer"
FEATKEY_SOP_INSTANCE_UID = "sop_instance_uid"
FEATKEY_STUDY_INSTANCE_UID = "study_instance_uid"
FEATKEY_ANNOTATION_STATUS = "annotation_status"

FEATKEY_COORDINATES_SEGMENTATION_MASK = "coordinates_segmentation_mask"
FEATKEY_COORDINATES_PECTORAL_MUSCLE_PARAMETERS = "coordinates_pectoral_muscle_parameters"
FEATKEY_COORDINATES_SKINLINE = "coordinates_skinline"
FEATKEY_COORDINATES_MAMMILLA_LOCATION = "coordinates_mammilla_location"

FEATKEY_PATCH_ID = "patch_id"
FEATKEY_PATCH_BOUNDING_BOX_CENTER_ROW = "patch_bounding_box_center_row"
FEATKEY_PATCH_BOUNDING_BOX_CENTER_COLUMN = "patch_bounding_box_center_column"
FEATKEY_PATCH_BOUNDING_BOX_WIDTH = "patch_bounding_box_width"
FEATKEY_PATCH_BOUNDING_BOX_HEIGHT = "patch_bounding_box_height"

# Label keys
LABELKEY_GT_ROIS_WITH_FINDING_CODES = "gt_rois_with_finding_codes"
LABELKEY_GT_ROIS_WITH_LABELS = "gt_rois_with_labels"
LABELKEY_GT_ANNOTATION_CONFIDENCES = "gt_annotation_confidences"
LABELKEY_GT_BIOPSY_PROVEN = "gt_biopsy_proven"
LABELKEY_PATCH_CLASS = "patch_class"
LABELKEY_FINDING_CODE = "gt_finding_code"
LABELKEY_GT_BIRADS_SCORE = "gt_birads_score"
# track here all region level label keys
LABELKEYS_REGION_LEVEL = {
    LABELKEY_GT_ROIS_WITH_FINDING_CODES,
    LABELKEY_GT_ROIS_WITH_LABELS,
    LABELKEY_GT_ANNOTATION_CONFIDENCES,
    LABELKEY_GT_BIOPSY_PROVEN,
    LABELKEY_GT_BIRADS_SCORE,
}

LABELKEY_DENSITY = "density"
LABELKEY_DENSITY_0_BASED = "density_0_based"
LABELKEY_ANNOTATION_ID = "annotation_id"
LABELKEY_ANNOTATOR_MAIL = "annotator_mail"
LABELKEY_WAS_REFERRED = "was_referred"
LABELKEY_WAS_REFERRED_IN_FOLLOWUP = "was_referred_in_followup"
LABELKEY_BIOPSY_SCORE = "biopsy_score"
LABELKEY_BIOPSY_SCORE_OF_FOLLOWUP = "biopsy_score_of_followup"
LABELKEY_INTERVAL_TYPE = "interval_type"

# Model output keys
OUTKEY_RCNN_ROIS = "rcnn_rois"
OUTKEY_RPN_ROIS = "rpn_rois"
OUTKEY_RPN_ROIS_FG_PROBS = "rpn_rois_fg_probs"

OUTKEY_GT_ROIS = "gt_rois"
OUTKEY_RCNN_CLASS_LABELS = "rcnn_class_labels"
OUTKEY_IMAGE_LEVEL_PROBS = "image_level_probs"
OUTKEY_PROB_MAP = "image_prob_map"
OUTKEY_RCNN_PROBS = "rcnn_probs"
OUTKEY_IMAGE_LEVEL_LABEL = "image_level_label"
OUTKEY_ANCHORS = "anchors"

