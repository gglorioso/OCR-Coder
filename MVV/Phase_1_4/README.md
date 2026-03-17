# Phase 1.4 — Micro-Texture & Indentation Staircase Perception

**Question Posed:** Can the model perceive the micro-texture of code syntax—like the depth of the indentation "staircase", or the thick/thin density of keywords—even when the actual text is too blurry to read?

**Finding: Yes, brilliantly, but only if the image is strictly preserved.** The model could perfectly feel the 'macro indentation staircase' (74.9% accuracy) and the rhythm of grayscale keyword density (R^2 = 0.690) by recognizing the contrast banding. However, we discovered that if you distort the aspect ratio of the image to squeeze it into a standard VLM square (e.g., squashing it to 768x768), that fragile micro-texture rhythm is physically destroyed, and accuracy plummets. Aspect-ratio must be preserved natively for Code-VLMs to function.
