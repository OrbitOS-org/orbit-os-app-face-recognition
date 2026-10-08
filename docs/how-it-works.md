# How Edge AI – Face Recognition works

Technical notes for developers: the models, how well they tell people apart, what happens to each camera frame, and where things are in the code. For what the app does and how to install it, see the [README](../README.md).

## Models

The app ships with two models, in [`cmd/face_recognition_client/orb/data/`](../cmd/face_recognition_client/orb/data/). Orbit Studio packages everything under `orb/data/` into the `.orb`.

| Model | File | What it does |
|---|---|---|
| **BlazeFace** (full range), from [MediaPipe](https://github.com/google-ai-edge/mediapipe) — TensorFlow Lite, Apache-2.0 | `blaze_face_full_range.tflite` | finds the faces in the frame and, for each, the eyes, the nose and the mouth |
| **SFace**, from the [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) — ONNX, quantised version (10 MB instead of 39 MB), Apache-2.0 | `face_recognition_sface_2021dec_int8.onnx` | turns a face into a list of 128 numbers (an embedding) that can be compared with others |

### How well it tells people apart

Measured with these two models and the app's own code for preparing the pictures (detection decoding, alignment), on 1,000 pairs of the public [LFW](http://vis-www.cs.umass.edu/lfw/) set — half of them two pictures of the same person, half two different people:

| Match threshold | Different people given a name | Same person recognised |
|---|---|---|
| 30% | 0.6% | 99.6% |
| **40% (default)** | **none of 498** | **98.2%** |
| 50% | none of 498 | 94.0% |

Two pictures of the same person scored 66% on average; two different people, 8% (the highest was 32%). The figures above are for the full-size model; on the whole LFW list (5,970 pairs) the full-size model was right for 99.50% of the pairs and the quantised one shipped here for 99.47%. These are pictures taken on different occasions; in front of a fixed camera the same person usually scores higher. People who look alike, such as close relatives, can score higher than strangers do — enroll them both, so that the app can tell them apart, and raise the threshold if needed.

## What happens to each frame

For each camera frame:

1. **Detect** — the frame is scaled to the 192×192 input of BlazeFace, keeping its proportions (black bars fill the rest). The model returns a box and keypoints for every face.
2. **Follow** — each face keeps its number from one frame to the next, so the app knows which faces it has already identified.
3. **Align** — the face is rotated and scaled so that the eyes, the nose and the mouth land on fixed positions.
4. **Embed** — SFace turns the aligned face into an embedding.
5. **Match** — the embedding is compared (cosine similarity) with every recording of every enrolled person; a person counts with the closest one. A name is shown only when the best match is above the threshold and clearly ahead of the second best person; otherwise the face is *Unknown*. While only one person is enrolled there is nobody to tell that person apart from, so the threshold is higher.

Steps 3 to 5 are the costly part and run only when needed: when a face appears, about once a second after that (more often while it is still unknown), and during an enrollment. With *Identify people* off, only step 1 runs.

The page shows the camera frames as they arrive and draws the boxes itself, from the results of the last frame analysed — the device does not redraw or recompress the video.

Each recording is stored as the average of the embeddings taken during it, not as pictures. When an update changes the model or how a face is prepared, earlier recordings can no longer be compared: the person stays in the list, marked to be recorded again.

| File | What |
|---|---|
| `main.go` | start-up |
| `config.go` | defaults and settings (camera, models, thresholds) |
| `face_client.go` | camera stream, frame loop, enrollment |
| `pipeline.go` | detection, face following, alignment and embedding; `pipeline_test.go` tests them without a device |
| `recognition.go` | matching an embedding against the enrolled people |
| `embeddings_store.go` | the enrolled people and their recordings, saved as a JSON file |
| `settings.go` | the match threshold chosen on the page |
| `webui.go`, `web/static/` | web page and its HTTP API |
