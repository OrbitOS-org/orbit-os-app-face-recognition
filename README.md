<p align="center">
  <img src="https://www.orbit-os.org/images/vscode/orbit-os-logo.png" width="300" alt="Orbit OS">
</p>

<h1 align="center">Edge AI – Face Recognition for Orbit OS</h1>

<p align="center"><b>Detect and recognise faces from a camera, live, on the device — an example of the Orbit OS Camera and AI APIs working together.</b></p>

An [Orbit OS](https://www.orbit-os.org/?ref=github-face-recognition) app that watches a camera, finds the faces in each frame and tells you who they are. Enroll a person by typing a name and recording their face for five seconds; from then on the app shows the name next to the face, live in the browser. Everything runs **locally on the device** through the Orbit OS AI service — no cloud API, and the faces you enroll stay on the device.

It is also a complete, readable example of **Edge AI on Orbit OS**: a camera stream, two models chained together (one TensorFlow Lite, one ONNX) and a web page, built with the [Orbit OS Go SDK](https://github.com/OrbitOS-org/orbit-os-sdk-go).

Runs on Raspberry Pi 3 / 4 / 5 and Arduino UNO Q with Orbit OS (free Community Edition) and a USB camera.

<p align="center">
  <img src="docs/store-screenshots/desktop-1-live.png" width="720" alt="Edge AI – Face Recognition: live view with a recognised person, the name and the match score on the face">
</p>

## Features

- **Live view** — the camera in your browser, with a frame and a name on each face
- **Enroll in seconds** — type a name, press *Start recording* and look at the camera for five seconds
- **Several recordings for each person** — record the same name again in other light, so that recognition is more reliable
- **Match threshold on the page** — you choose how strict recognition is
- **Everything on the device** — no cloud service; frames are not stored and nothing leaves the device
- **Works right after install** — the models come with the app
- **Manage enrolled people**, choose the camera, pause it; works on a phone too

<p align="center">
  <img src="docs/store-screenshots/desktop-2-enroll.png" width="720" alt="Edge AI – Face Recognition: recording a person in the Enroll section, with the list of enrolled people">
</p>

## Install

**From the Orbit OS Store (recommended):** install [Edge AI – Face Recognition](https://store.orbit-os.org/app/face-recognition-client?ref=github-face-recognition) on your device in one click.

<a href="https://store.orbit-os.org/app/face-recognition-client?ref=github-face-recognition"><img src="https://www.orbit-os.org/images/badges/get-it-on-orbit-os-store@3x.png" width="200" alt="Get it on Orbit OS Store"></a>

**From source — recommended: [Orbit Studio](https://marketplace.visualstudio.com/items?itemName=orbit-os.orbit-studio) (VS Code):**

You need [VS Code](https://code.visualstudio.com/) with the Orbit Studio extension and **[Go](https://go.dev/dl/) 1.25 or newer** installed (`go` on your PATH).

1. Clone the repository and open the folder in VS Code with the Orbit Studio extension:
   ```bash
   git clone https://github.com/OrbitOS-org/orbit-os-app-face-recognition
   code orbit-os-app-face-recognition
   ```
2. In the Orbit sidebar, run **Add / Update SDK** and set your device's IP.
3. Use **Run** to try it live against a device in Developer Mode, then **Build + Deploy** to install the signed `.orb`.

**Without Orbit Studio:** unpack the [SDK release](https://github.com/OrbitOS-org/orbit-os-sdk-go/releases/tag/v26.0.3) into `orbit-os-sdk-go/`, then `go build ./cmd/face_recognition_client` builds the binary — use Orbit Studio to package and sign the `.orb`.

## Getting started

1. Connect a USB camera to the device.
2. Open **Edge AI – Face Recognition** from the AppHub on your device. The live view starts and faces are framed as they appear.
3. Open **Enroll**, type the person's name, press **Start recording** and keep the face in view for five seconds.
4. Back in **Live**, the name and the match score appear next to the face.

## How it works

Two models run on the device through the Orbit OS AI service. **BlazeFace**, from [MediaPipe](https://github.com/google-ai-edge/mediapipe), finds the faces in each camera frame. **SFace**, from the [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface), turns each face into a list of numbers that is compared with the people you enrolled. Both are Apache-2.0 and ship inside the app.

Measured on 1,000 pairs of the public LFW set with the app's own code, at the default threshold no pair of different people was taken for the same person, and the same person was recognised in 98% of the pairs.

The details — the models, the measurement, what happens to each frame and the code map — are in [docs/how-it-works.md](docs/how-it-works.md).

## Development (Orbit Studio)

This project follows the standard [Orbit Studio](https://marketplace.visualstudio.com/items?itemName=orbit-os.orbit-studio) layout:

| Path | What |
|---|---|
| `cmd/face_recognition_client/` | app source, `metadata.json` (manifest & permissions) |
| `cmd/face_recognition_client/orb/icon.svg` | launcher / Store icon |
| `cmd/face_recognition_client/orb/data/` | the models packaged into the `.orb` |
| `orbit.project.json` | Orbit Studio project settings |

- **Recommended workflow:** open the folder in VS Code with Orbit Studio, **Add / Update SDK** (downloads the SDK into `orbit-os-sdk-go/`, which is not in the repository), then **Run** to develop against a device in Developer Mode, or **Build + Deploy** to install the `.orb`.
- The project builds against that local SDK copy (`go.mod` and `go.work` point to it).
- Development TLS certificates live in `cmd/certs/grpc/` and are never committed.
- `go test ./cmd/face_recognition_client` runs the tests of the pipeline, the matching and the people file (no device needed).
- Permissions used: `CameraService`, `AiService`, `AppHubService`, `SystemService`.

## Privacy and security

- **Everything stays on the device.** Camera frames are processed locally and are not stored. Enrolled people are saved as a name and embeddings in `face_db/embeddings.json`, in the app's data folder on the device, readable by the app only; no pictures are kept, and the embeddings are never sent to the browser.
- **Face data is personal data.** Enroll only people who agreed to it, and follow the rules that apply where you use the app.
- The page listens on `127.0.0.1` only and is reached through the Orbit OS AppHub at `http://<DEVICE_IP>/face-recognition`, behind the device login. It is not reachable directly from the network.
- The app takes a port in the reserved range 50000–60000 (starting at 50004) and moves to the next one when a port is taken. If it cannot open its page it stops, instead of running without one.
- When the app is stopped it takes its page off the AppHub and frees the camera.

## Links

[App in the Store](https://store.orbit-os.org/app/face-recognition-client?ref=github-face-recognition) · [Orbit OS](https://www.orbit-os.org/?ref=github-face-recognition) · [Getting started](https://www.orbit-os.org/getting_started.html?ref=github-face-recognition) · [SDK reference](https://www.orbit-os.org/api-reference.html?ref=github-face-recognition) · [Forum](https://forum.orbit-os.org/?ref=github-face-recognition) · info@orbit-os.org

## Acknowledgments

Face detection uses the **BlazeFace** model from **[MediaPipe](https://github.com/google-ai-edge/mediapipe)** by Google. Face recognition uses **[SFace](https://github.com/zhongyy/SFace)** by Yaoyao Zhong and co-authors, in the ONNX version published in the **[OpenCV Zoo](https://github.com/opencv/opencv_zoo)**. Thanks to both teams for publishing them.

## License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
