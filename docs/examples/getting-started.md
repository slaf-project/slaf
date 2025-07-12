# Getting Started Examples

## Interactive Example

<iframe src="../01-getting-started.html" width="100%" height="800px" style="border:1px solid #ccc; border-radius:8px;"></iframe>

This page shows how to work with the SLAF getting started examples.

## Marimo Notebooks

The examples in this project are written as [Marimo](https://marimo.io/) notebooks. Marimo provides an excellent interactive environment for data science and machine learning workflows.

## Viewing Examples

### Interactive Mode

To run the examples interactively:

```bash
# Install marimo if not already installed
pip install marimo

# Run the getting started notebook
marimo edit examples/01-getting-started.py
```

This will open an interactive Marimo editor where you can run cells, modify code, and explore the data.

### Static HTML Export

Marimo provides built-in HTML export functionality. You can export any notebook to static HTML:

#### From the Marimo Editor

1. Open the notebook in Marimo editor
2. Click the menu button (three dots) in the top-right
3. Select "Export to static HTML"
4. Save the HTML file

#### From Command Line

```bash
# Export a notebook to HTML
marimo export html examples/01-getting-started.py -o examples/getting-started.html
```

#### Programmatic Export

You can also export notebooks programmatically:

```python
import marimo

# Export notebook to HTML
marimo.export_html("examples/01-getting-started.py", "examples/getting-started.html")
```

## Available Examples

### 01-getting-started.py

A comprehensive introduction to SLAF covering:

- Loading SLAF datasets
- Understanding the database schema
- Running SQL queries
- Filtering cells and genes
- Getting expression data
- Scanpy integration
- Performance characteristics

### 02-lazy-processing.py

Demonstrates lazy evaluation and processing:

- Lazy data loading
- Memory-efficient operations
- Batch processing
- Performance optimization

### 03-ml-training-pipeline.py

Shows ML training workflows:

- Data preprocessing
- Tokenization
- Model training
- Evaluation

## Embedding Examples in Documentation

To include these examples in your documentation:

1. Export the notebooks to HTML using Marimo's built-in export
2. Place the HTML files in your documentation directory
3. Include them using iframes or embed them directly

### Example Embedding

```html
<iframe src="01-getting-started.html" width="100%" height="800px"></iframe>
```

## Customizing Examples

You can customize the examples for your specific use case:

1. **Modify data paths**: Update the paths to point to your SLAF datasets
2. **Adjust parameters**: Change filtering criteria, query parameters, etc.
3. **Add visualizations**: Include plots and charts specific to your analysis
4. **Extend functionality**: Add new cells with additional analysis

## Best Practices

### For Interactive Use

- Use descriptive cell names
- Include markdown cells for explanations
- Add progress indicators for long-running operations
- Use the variables panel to explore data

### For Documentation

- Keep examples focused and concise
- Include clear explanations in markdown cells
- Use consistent formatting
- Test examples with different datasets

### For Export

- Ensure all dependencies are available
- Test the exported HTML in different browsers
- Optimize for readability in static format
- Include navigation if exporting multiple notebooks

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed
2. **Data not found**: Check file paths and dataset availability
3. **Memory issues**: Use lazy evaluation for large datasets
4. **Export problems**: Verify Marimo version and export options

### Getting Help

- Check the [Marimo documentation](https://docs.marimo.io/)
- Review the [SLAF API reference](../api/core.md)
- Open an issue on GitHub for specific problems

## Next Steps

- Explore the [User Guide](../user-guide/how-slaf-works.md) for detailed concepts
- Check the [API Reference](../api/core.md) for complete documentation
- Try the [other examples](lazy-processing.md) for advanced usage
