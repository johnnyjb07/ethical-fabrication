# Using Machine Learning to Identify Potential Firearm Components in 3D Printing Files

## Abstract
The rapid expansion of 3D printing has enabled near unlimited access to personal manufacturing while also introducing serious security risks, including the unregulated creation of firearm components. Existing safeguards such as keyword filtering and DRM takedowns are insufficient to prevent the creation and distribution of firearm-related designs. This paper investigates the feasibility of using machine learning to automatically classify 3D-printable models as firearm-related or non-firearm-related. Drawing on prior work in 3D object classification, the proposed approach explores convolutional neural network–based methods applied to learning features on a model, and for those features to be inputs on a support vector machine classifier. The study aims to develop a classification model, evaluate its accuracy and limitations, and assess its potential for scalable deployment across 3D printing platforms. In addition to technical performance, the paper examines ethical, regulatory, and privacy considerations associated with automated detection systems. Key challenges include dataset limitations, labeling bias, and ambiguity in classifying partial or visually similar components. Overall, this research contributes to emerging efforts to improve safety and content moderation in the 3D printing ecosystem.

## Background and Motivation
The widespread adoption of 3D printing technology has created new opportunities for personal manufacturing while simultaneously introducing significant security vulnerabilities. This technology enables individuals to produce useful items for daily life, but it also allows bad actors to manufacture dangerous items, including functional firearms. This presents a serious problem: individuals can now access potentially lethal weapons at low cost while completely bypassing established safety measures such as background checks and licensing requirements that govern traditional firearm purchases.

However, a potential solution exists to address this growing threat. Just as ink and laser printers incorporate anti-counterfeiting measures that prevent users from scanning, printing, or copying currency, a similar technology for 3D printers could be developed. Such a system would analyze designs before printing and detect whether components being manufactured bear too close a resemblance to firearm parts, effectively preventing the production of weapon components. With a growing criminal industry printing thousands of gun components a day, automated and scalable detection tools become a much-needed part of the 3D printing environment.

There is currently no widely adopted automated method for identifying 3D-printable designs that may be associated with firearm components, which poses challenges for safety and content moderation. Methods in place right now include keyword blocking and DRM takedowns, but not much else is currently regulating the space. This puts the general populus at risk for getting harmed by an illegally printed firearm and puts extra strain on law enforcement to take these weapons off the street.

## Research Questions and Objectives
Questions to explore:
- Can a machine learning model classify 3D-printable designs as firearm-related or non-firearm-related?
- How accurate is the model?
- Is such a model feasible for deployment across services?

Objectives:
- Build a simple classification model for 3D-printable objects.
- Evaluate its accuracy and limitations.
- Explore ethical, regulatory, and privacy concerns related to automated detection.

## Literature Review
Many classification models exist, with convolutional neural networks (CNNs) achieving success in 2D image classification. Shao and Xu [1] explore using CNNs to classify 3D models. They use voxelization, which divides the workspace into voxels (cubes) labeled 1 if occupied and 0 if empty. “When the classification object expands from 2D to 3D, transformation capacity of the object gets larger because the rotation could be done in 3D spaces” [1]. To handle rotations and translations, they employ a spatial transformer layer for alignment.

Qin, Li, and Gao [2] propose selecting and extracting common features, preprocessing them (e.g., via a spatial transformer layer), constructing a classifier by allocating neurons per hidden layer, and training on processed feature patterns. Another approach is two-step: first a CNN learns features from the original data, then the learned features serve as input vectors to train a support vector machine classifier.

## Methodology
This project uses a computational and experimental design to develop and evaluate an agentic AI that classifies 3D models as firearm-related or non-firearm-related. The model will be trained on labeled examples of 3D models from public repositories hosting STL or OBJ files. The dataset will be split into two main categories: firearm component and not a firearm component. Firearm components can be further subcategorized by gun type and specific part.

Models will be sliced, with slices stored in short-term memory; if adjacent slices begin to resemble trained firearm parts, the system raises a suspicion score until a threshold triggers a “not allowed” classification.

## Ethical Considerations
The research focuses on classification and detection of potentially unethical 3D-printable models. All data will be sourced from publicly available repositories.

## Limitations
Limitations include dataset size, potential labeling bias, and difficulty distinguishing partial firearm components from visually similar non-weapon components. Model uncertainty is another concern—e.g., how to act when the model is only 60% certain about an object.