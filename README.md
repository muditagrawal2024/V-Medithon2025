# V-Medithon2025
This is the Sample CV program for the SmartDrops project.

**One-line description:**
A handheld, AI-assisted eye drop dispenser that uses a camera, computer vision, and on-device voice guidance to align, dispense a precise micro-dose, verify success, and log left/right eye adherence.

**Problem it solves:**
Reduces medication waste by dispensing a single, controlled drop only when alignment is correct and the eye is open.
Improves adherence with reminders, per-eye counts, and tamper-resistant verification (prevents “fake” administration).
Increases usability for seniors and low-vision users with real-time voice guidance and simple HUD overlays.

**What the device is:**
Handheld “smart pen” that avoids bulky stands or rigs, making daily use practical and portable.
Integrated camera for real-time eye detection, alignment, and success verification.
Microphone + speaker for clear, step-by-step voice instructions and feedback.
IMU-ready design (gyroscope/accelerometer) for stability and angle checks; simulated in the hackathon demo for feasibility.

**How it works (user flow):**
Detect face and identify target eye (left/right) with facial landmarks.
Guide the user to align the dropper using directional voice prompts (move left/right/up/down) and on-screen hints.
Confirm steady alignment for a short, configurable hold time.
Automatically trigger a single drop when the eye is open and the dropper is correctly positioned.
Verify success using overlap-based computer vision; if not confirmed within a timeout, prompt retry.
Log the event with timestamp, eye side, and success/failure, then proceed to the next eye as needed.

**What has been built already:**
Computer vision pipeline using facial landmarks (MediaPipe-style indices) to robustly track both eyes and calculate stable eye centers.
Left/right eye differentiation and eye-open status with smoothing (EMA) to reduce jitter and false transitions.
Custom object detection for the dropper using a trained model; simple overlap (IoU) confirmation between the detected dropper and an eye ROI.
A finite state machine per eye: WAIT_FOR_FACE → ALIGN → READY → DISPENSE → VERIFY → NEXT EYE, ensuring safe, predictable behavior.
Non-blocking, queued text-to-speech with cooldown to avoid voice spam and keep the UI responsive.
HUD overlays: clean white headers with black text, detection markers, and current state/eye targeting indicators.
CSV logging of administrations, including timestamp, date, eye, and success—supporting adherence review and auditability.

**Why this approach is robust:**
Alignment assurance: The system won’t dispense until the dropper is within a pixel threshold of the eye center for consecutive frames and the device is steady.
Open-eye gating: Dispense triggers only when the eye is detected open, reducing misfires.
Anti-faking features: The camera verifies dropper-eye overlap for a minimum time window, discouraging false positives (“white coat” behavior).
Hawthorne effect leverage: Knowing administrations are observed and recorded increases real-world adherence, improving outcomes.

**What makes it user-friendly:**
Fully handheld—no need for stands or complex setups.
Real-time voice instructions for alignment and timing (eyes open, hold still, dispensing now, next eye).
Simple on-screen cues for users with some vision and audio-first prompts for low-vision users.

**Data and adherence features:**
Per-eye counters and daily targets aligned with prescriptions.
Timestamped logs for clinicians to assess adherence patterns, dosing intervals, and success rate.
Extensible to reminders, provider dashboards, and EHR integration in future iterations.

**What’s next after the hackathon:**
Hardware IMU integration to replace simulation and add stability/angle thresholds.
Micro-dosing nozzle to minimize waste and match ocular absorption capacity.
Stronger drop-verification using motion/optical flow and liquid-splash cues for higher confidence.
On-device safety checks (distance estimation, eyelid margin proximity) for added protection.
Optional cloud sync and clinician portal for remote monitoring and dosage personalization.

**Impact highlights:**
Cuts drop waste by dispensing only when aligned and eye is open, saving costly medications.
Improves adherence via objective tracking and prompts, addressing a major failure point in glaucoma care.
Increases independence and accuracy for patients with tremor, arthritis, or low vision.
