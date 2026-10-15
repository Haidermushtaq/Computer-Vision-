# Lab 06 Answers

## Task 1: Landmark image database

The reference database contains five landmark classes: Eiffel Tower, Taj Mahal, Great Wall of China, Statue of Liberty, and Colosseum. Each class has five reference images, organized in a separate directory under `landmark_project/landmark_db/`.

## Task 2: SIFT feature extraction

The supplied project includes the reference images and a precomputed `sift_database.pkl`. The associated notebook is intended to load reference images, convert them to grayscale, extract SIFT keypoints and descriptors, associate each descriptor set with its landmark label, and visualize reference keypoints.

## Task 3: Query processing

The query set is arranged under `landmark_project/landmark_tests/`. It includes original, rotated, resized, reduced-brightness, and different-viewpoint examples for each known landmark, plus two images of landmarks absent from the database. The saved output folder contains query keypoint and matching visualizations.

## Task 4: Matching and recognition

The supplied `landmark_project/outputs/results_table.csv` records the best candidate, keypoint count, reliable-match count, decision, and predicted label for 27 queries. The system rejects low-confidence candidates rather than accepting the largest match count by itself. In these saved results, all five accepted queries were Taj Mahal images and were correct; the other 20 known-landmark queries were rejected. Both unknown-landmark queries were rejected.

## Task 5: Landmark localization

The project outputs include matching and localization visualizations. The intended geometric check estimates a homography with RANSAC from reliable reference/query correspondences, projects the four reference-image corners into the query, and draws a region only when the transform is reliable.

## Task 6: Performance evaluation

The supplied table evaluates 25 known-landmark queries across the five requested conditions and two unknown-landmark queries. Five of 25 known queries were recognized correctly (20% known-query recognition); both unknown queries were rejected. Seven of all 27 rows have `Correct=True` because the two unknown rejections are also correct decisions. The mean recorded processing time is 2.95 seconds per query. The full per-query values are in `landmark_project/outputs/results_table.csv`.
