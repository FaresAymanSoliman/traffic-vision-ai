# TrafficVision AI Requirements

## 1. Document Information

- Project: TrafficVision AI
- Project type: Full-stack AI traffic and road monitoring system
- Development model: Solo development
- Primary input for the MVP: Uploaded MP4 traffic video
- Primary frontend: React with TypeScript
- Primary backend: FastAPI with Python
- Computer vision stack: YOLO, ByteTrack, and OpenCV

## 2. Problem Statement

Traffic monitoring commonly requires human operators to observe video streams,
count vehicles, identify congestion, and recognize potential traffic violations.

Manual monitoring is time-consuming, difficult to scale, and vulnerable to human
error. TrafficVision AI will process traffic videos automatically and produce
structured information about vehicles, movement, speed, traffic density, and
potential violations.

The system is intended as an analytical and educational prototype. Its automated
results must be treated as estimates that can require human review.

## 3. Project Objectives

TrafficVision AI will:

1. Detect relevant vehicles in traffic videos.
2. Track vehicles across consecutive frames.
3. Count vehicles without counting the same vehicle repeatedly.
4. Classify vehicles by type.
5. Estimate vehicle speed after camera calibration.
6. classify traffic density.
7. Detect selected potential traffic violations.
8. Store processing results and events.
9. Present results through a responsive web dashboard.
10. Support uploaded videos before introducing live streams.

## 4. Target Users

The initial target users are:

- Project developers and computer vision students
- Traffic-analysis researchers
- Traffic monitoring operators
- Academic project reviewers

The initial version is not intended to issue official traffic penalties or make
fully automated legal decisions.

## 5. MVP Scope

The first full-stack MVP will support:

- Uploading one MP4 traffic video
- Creating a video-processing job
- Detecting vehicles with YOLO
- Tracking vehicles with persistent IDs
- Counting vehicles crossing one configured line
- Reporting counts by vehicle class
- Producing an annotated output video
- Saving processing results
- Displaying results in a React dashboard
- Showing processing progress
- Showing basic charts and summary cards

## 6. Supported Vehicle Classes

The initial supported classes are:

- Car
- Truck
- Bus
- Motorcycle

Additional classes may be added after the initial classes work reliably.

## 7. Functional Requirements

### FR-001: Video upload

The system shall allow a user to upload an MP4 traffic video.

### FR-002: File validation

The system shall validate the uploaded file's format and configured maximum size.

### FR-003: Processing jobs

The system shall create a processing job for each accepted video.

### FR-004: Job status

The system shall expose the processing job's status and progress.

Supported statuses shall include:

- Pending
- Processing
- Completed
- Failed
- Cancelled

### FR-005: Vehicle detection

The system shall detect supported vehicle classes in video frames.

Each detection shall contain:

- Vehicle class
- Confidence score
- Bounding-box coordinates
- Frame number
- Video timestamp

### FR-006: Vehicle tracking

The system shall assign a persistent tracking ID to a detected vehicle across
consecutive frames when possible.

### FR-007: Vehicle counting

The system shall count a tracked vehicle when it crosses a configured counting line.

A tracked vehicle shall not be counted more than once for the same line and movement
direction.

### FR-008: Per-class counts

The system shall produce total vehicle counts grouped by vehicle class.

### FR-009: Direction counts

The system shall support separate counts for configured movement directions.

### FR-010: Annotated video

The system shall generate an output video containing:

- Vehicle bounding boxes
- Vehicle class labels
- Confidence scores
- Tracking IDs
- Counting line
- Current vehicle counts

### FR-011: Speed estimation

After camera calibration, the system shall estimate vehicle speeds using tracked
positions and time differences.

Speed values shall be presented as estimated speeds.

### FR-012: Traffic density

The system shall classify traffic density as:

- Low
- Medium
- High

Density thresholds shall be configurable for each camera or video configuration.

### FR-013: Violation events

The system shall support potential violation events generated from tracked
vehicle behavior.

The initial planned violations are:

- Wrong-way movement
- Restricted-zone entry
- Stopped vehicle
- Estimated speeding

Violations shall be developed one at a time after the tracking system is stable.

### FR-014: Evidence

A confirmed violation event shall store an evidence image or video reference.

### FR-015: Human review

Violation records shall support review statuses:

- Unreviewed
- Confirmed
- Dismissed

### FR-016: Results dashboard

The dashboard shall display:

- Total vehicle count
- Count by vehicle class
- Average estimated speed
- Traffic density
- Number of violation events
- Processing status
- Output video
- Traffic charts

### FR-017: Results persistence

The system shall store processing jobs, analytics summaries, and violation events
in a database.

### FR-018: Live updates

The backend shall be able to send processing-progress and analytics updates to
the frontend.

WebSockets will be introduced after the normal REST-based processing flow works.

## 8. Non-Functional Requirements

### NFR-001: Maintainability

Detection, tracking, counting, calibration, speed estimation, density estimation,
and violation detection shall be implemented as separate modules.

### NFR-002: Configurability

The following values shall not be permanently hard-coded:

- Confidence thresholds
- Counting-line coordinates
- Road-zone coordinates
- Density thresholds
- Speed limits
- Violation thresholds
- Model paths

### NFR-003: Reproducibility

Model training shall record:

- Dataset version
- Data split
- Model version
- Training configuration
- Evaluation results
- Random seed when applicable

### NFR-004: Reliability

A failed processing job shall store a useful error message without causing the
entire API to terminate.

### NFR-005: Security

The application shall:

- Validate file extensions and content types
- Limit uploaded file sizes
- Sanitize filenames
- Protect secrets with environment variables
- Restrict configured cross-origin requests
- Avoid exposing internal filesystem paths

### NFR-006: Privacy

The initial system shall not perform facial recognition or identify individual
people.

### NFR-007: Explainability

A violation event shall include its type, timestamp, involved tracking ID,
supporting measurements, and evidence reference.

### NFR-008: Performance monitoring

The vision worker shall report relevant performance information such as:

- Video FPS
- Processing FPS
- Average inference time
- Number of processed frames
- Number of dropped frames for real-time processing

### NFR-009: Testability

Core geometry, counting, speed, density, and violation rules shall be testable
without starting the web server or loading the complete frontend.

### NFR-010: Portability

The project shall support containerized execution using Docker.

## 9. Initial Success Metrics

The initial targets are engineering goals, not final guarantees.

### Detection

- Measure per-class precision and recall.
- Record model mAP on the selected test dataset.
- Compare pretrained and fine-tuned model performance.

### Tracking

- Measure identity stability on selected test clips.
- Document visible ID switches and lost tracks.

### Counting

- Target no more than 10 percent absolute counting error on the selected test videos.
- Report counting accuracy separately for each vehicle class.

### Speed

- Validate estimated speed using footage with known distance or reference speed.
- Report mean absolute speed error.
- Do not claim validated speed accuracy before calibration testing.

### Violations

- Measure precision and recall for each implemented violation independently.
- Require events to persist for multiple observations before confirmation.

### Performance

- Report processing FPS and inference latency on the development computer.
- Optimize only after establishing baseline measurements.

## 10. MVP User Flow

1. The user opens the React application.
2. The user uploads an MP4 traffic video.
3. The FastAPI backend validates and stores the video.
4. The backend creates a processing job.
5. The vision worker reads the video.
6. YOLO detects vehicles.
7. ByteTrack assigns persistent vehicle IDs.
8. The counting module generates crossing events.
9. The system writes an annotated video.
10. Results are stored in the database.
11. The frontend displays the completed results.

## 11. Out of Scope for the First MVP

The first MVP will not include:

- Facial recognition
- Driver identification
- Automatic traffic penalties
- License plate recognition
- Multiple simultaneous RTSP cameras
- Traffic-light recognition
- Red-light violation detection
- Cloud autoscaling
- Mobile applications
- Complex user roles
- Distributed microservices
- Guaranteed legal-grade speed measurement
- Advanced model optimization with TensorRT

These features may be evaluated after the uploaded-video MVP is complete.

## 12. Assumptions

- The first input will be a recorded MP4 video.
- The road and relevant vehicles will be visible.
- The video will have valid frame-rate metadata.
- The camera will be mostly stationary for counting and speed analysis.
- Speed estimation will require camera-specific calibration.
- Density thresholds may differ between roads and cameras.
- AI-generated events may contain false positives and false negatives.

## 13. Technical Constraints

- Model performance depends on the available CPU, GPU, and memory.
- Real-time processing may not be possible on every computer.
- Occlusion can interrupt vehicle tracking.
- Low resolution, darkness, rain, and camera motion may reduce accuracy.
- Perspective distortion prevents accurate speed estimation without calibration.
- Pretrained model classes may not perfectly match the final target data.
- Dataset and model licenses must be reviewed before redistribution or commercial use.

## 14. Future Scope

Possible features after the first stable release include:

- RTSP camera support
- Multiple cameras
- Traffic-signal integration
- Red-light violation detection
- License plate detection
- Lane-specific analytics
- Queue-length estimation
- Incident detection
- Advanced model optimization
- Cloud deployment with GPU workers
- Notifications
- Role-based access control