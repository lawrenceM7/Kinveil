# Face recognition design

Kinveil treats face recognition as local identity estimation, not authentication.
A match may select a person's memories and preferences, but sensitive tools must
still pass through the permission system.

## Enrollment pipeline

1. Require explicit consent and a display name.
2. Sample frames from the enrollment video instead of retaining the video.
3. Require exactly one sufficiently large, confidently detected face.
4. Produce a normalized embedding with a pretrained InsightFace model.
5. Reject near-duplicate samples and store up to 24 varied templates in SQLite.
6. Delete all partial data if no usable frame is found.

The continuous camera path can also promote the one visible temporary visitor
after explicit biometric consent. Persistent templates are maintained as a
bounded representative pool: 24 samples per person, with no more than 8 in one
coarse head-pose bucket. New near-duplicates are ignored unless their detector
quality is meaningfully better; when a bucket is full, redundant or lower-
quality samples are pruned. Automatic updates for returning people require a
higher match confidence than ordinary recognition to reduce identity drift.

The video should slowly cover frontal, left, right, slightly up and slightly down
views under ordinary lighting. Raw frames are not stored by the current module.

## Recognition and rejection

Cosine similarity is calculated between a detected face and each enrolled
template. A result is accepted only if it clears both a similarity threshold and
a margin over the second-best identity. Thresholds must be calibrated on the
target camera and environment before any security-sensitive use.

## Why no custom CNN training

Training a classifier from a short user video plus a public face dataset creates
an expensive closed-set model: adding a person means retraining, public images
become negative-class baggage, and an unknown visitor is easily forced into a
known class. A pretrained metric-learning network already learns general facial
features; Kinveil only stores compact templates for each consenting person.

Before production use, add liveness/anti-spoofing, encrypted storage, retention
controls, camera-state indicators and a user-facing review/delete screen.
