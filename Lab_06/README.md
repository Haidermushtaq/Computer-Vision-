# Lab 06: Landmark Recognition Using SIFT

Haider Mushtaq (FA23-BAI-044)

This lab builds a landmark-recognition pipeline using SIFT keypoints and descriptors, brute-force k-nearest-neighbor matching, Lowe's ratio test, ranked candidate scores, a rejection threshold, and RANSAC homography localization.

## Folder contents

- `Landmark_Recognition_SIFT.ipynb`: Lab 06 notebook.
- `landmark_project/landmark_db/`: reference images for five known landmarks.
- `landmark_project/landmark_tests/`: transformed known-landmark queries and unknown landmark queries.
- `landmark_project/sift_database.pkl`: precomputed SIFT database supplied with the project.
- `landmark_project/outputs/`: saved keypoint, ranking, localization, demo figures, and `results_table.csv`.
- `REPORT.md`: completed report based on the supplied evaluation table and screenshots.
- `ANSWERS.md`: direct responses to the assignment's six implementation tasks.

Run the notebook with `Lab_06/` as its working directory. Its code uses `landmark_project/` as the project root, matching the supplied folder structure. The SIFT pickle is a Python-specific serialized artifact and should be loaded only from this trusted project.
