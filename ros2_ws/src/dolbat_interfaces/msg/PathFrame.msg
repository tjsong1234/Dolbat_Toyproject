# Source frame identity. *_bev_cm arrays use the calibrated BEV reference,
# not the rear axle. /dolbat/path_rear is the standard metre-valued rear path.
std_msgs/Header header
uint32 frame_index
float64 source_fps
bool end_of_stream
string status
string reason
string method
string perception_status
float64[] path_bev_cm
float64[] left_model_bev_cm
float64[] right_model_bev_cm
float64[] left_observed_rows_cm
float64[] right_observed_rows_cm
float64[3] left_coefficients
float64[3] right_coefficients
bool left_observed
bool right_observed
float64[] partial_left_cm
float64[] partial_right_cm
bool joint_fit_accepted
float64 max_path_adjustment_cm
