# Source frame identity for synchronization. LP coordinates below are in the
# assumed rear_axle frame: X forward, Y left, metres; steering is positive left.
# The separate PointStamped telemetry carries frame_id=rear_axle.
std_msgs/Header header
uint32 frame_index
float64 source_fps
bool end_of_stream
bool valid
string status
float64 target_x_m
float64 target_y_m
float64 steering_rad
float64 steering_deg
float64 curvature_per_m
float64 heading_error_rad
int32 segment_index
float64 segment_fraction
float64 circle_residual_m
uint32 candidate_count
float64 actual_lookahead_m
