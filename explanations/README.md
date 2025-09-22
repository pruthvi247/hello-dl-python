# CNN Explanations and Visual Demos

This folder contains visual demonstrations and educational materials for understanding Convolutional Neural Network (CNN) concepts.

## Files

### `translation_invariance_demo.py`

**Main visual demonstration of translation invariance in CNNs**

This comprehensive demo shows how shifting (translating) an input image affects different stages of CNN processing:

1. **Raw Image** - Highly sensitive to translation
2. **Convolution Output** - Feature detection but position-dependent
3. **Edge Detection** - Less sensitive to exact position
4. **Pooled Output** - Most translation invariant

**Features:**

- Side-by-side visualizations of 4 processing stages
- Multiple test patterns (cross, square, L-shape, circle)
- Quantitative similarity analysis using cosine similarity
- Interactive demo with multiple pattern options
- Clear visualization of how pooling provides translation invariance

**Usage:**

```bash
# Make sure you're in the project root and have activated the environment
source pytensor-env/bin/activate

# Run the interactive demo
python explanations/translation_invariance_demo.py
```

**Dependencies:**

- numpy
- matplotlib
- Pillow (PIL)
- pytensorlib

### `translation_invariance_quick_test.py`

**Simplified version for automated testing**

A non-interactive version that demonstrates the same concepts with a single pattern and generates a quick summary.

**Usage:**

```bash
python explanations/translation_invariance_quick_test.py
```

## Key Concepts Demonstrated

### Translation Invariance

The ability of a neural network to recognize features regardless of their position in the input. This is crucial for image recognition tasks where objects may appear at different locations.

### Processing Stages Analysis

1. **Raw Image Processing**

   - Direct pixel comparison
   - Highly sensitive to position changes
   - Low similarity scores when images are shifted

2. **Convolution Layer**

   - Detects local features using learned filters
   - Some sensitivity to position
   - Feature maps show where patterns are detected

3. **Edge Detection**

   - Uses Sobel-like filters to detect edges
   - Less position-sensitive than raw images
   - Highlights important structural features

4. **Max Pooling**
   - Reduces spatial dimensions while preserving important features
   - Provides the highest translation invariance
   - Enables recognition regardless of exact position

### Similarity Metrics

The demo uses **cosine similarity** to quantitatively measure how similar feature maps are between the original and translated images:

- **1.0** = Perfect similarity (identical features)
- **0.0** = No similarity
- **Negative values** = Opposing patterns

### Expected Results

Generally, you should observe:

- **Pooling similarity > Edge detection similarity > Convolution similarity**
- Higher pooling similarity scores indicate better translation invariance
- Improvement values show how much pooling helps vs. raw convolution

## Educational Value

This demo provides intuitive understanding of:

- Why CNNs use pooling layers
- How translation invariance is achieved
- The trade-offs between spatial precision and invariance
- Visual proof of CNN robustness to input variations

## Customization

You can modify the demo by:

- Changing image sizes in `TranslationInvarianceDemo(image_size=N)`
- Adding new test patterns in `create_test_image()`
- Adjusting translation amounts in the `shifts` parameter
- Modifying filters for different feature detection

## Technical Implementation

The demo includes:

- Pure NumPy implementation of convolution and pooling
- Multiple edge detection filters (Sobel vertical/horizontal)
- Feature detection filters for blob/corner detection
- Cosine similarity computation for quantitative analysis
- Matplotlib visualizations with proper scaling and colormaps
