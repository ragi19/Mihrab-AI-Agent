"""
Data visualization tools for generating charts and graphs
"""

import base64
import io
import json
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from ...utils.logging import get_logger
from ..base import BaseTool

logger = get_logger("tools.visualization")


class ChartGeneratorTool(BaseTool):
    """Tool for generating data visualizations"""

    def __init__(self):
        super().__init__(
            name="chart_generator", description="Generate charts and graphs from data"
        )
        self._parameters = {
            "chart_type": {
                "type": "string",
                "description": "Type of chart to generate",
                "enum": ["bar", "line", "pie", "scatter", "histogram"],
                "default": "bar",
            },
            "title": {"type": "string", "description": "Chart title"},
            "data": {
                "type": "object",
                "description": "Data for the chart. Format depends on chart type.",
            },
            "x_label": {"type": "string", "description": "Label for x-axis"},
            "y_label": {"type": "string", "description": "Label for y-axis"},
            "colors": {
                "type": "array",
                "description": "List of colors for chart elements",
                "items": {"type": "string"},
            },
            "width": {
                "type": "number",
                "description": "Width of the chart in inches",
                "default": 10,
            },
            "height": {
                "type": "number",
                "description": "Height of the chart in inches",
                "default": 6,
            },
            "dpi": {
                "type": "integer",
                "description": "Resolution of the chart",
                "default": 100,
            },
        }
        self._required_params = ["chart_type", "data"]
        logger.info("Initialized tool: chart_generator")

    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for tool parameters"""
        return {
            "type": "object",
            "properties": self._parameters,
            "required": self._required_params,
        }

    async def _execute(
        self,
        chart_type: str,
        data: Dict[str, Any],
        title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        colors: Optional[List[str]] = None,
        width: float = 10,
        height: float = 6,
        dpi: int = 100,
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute the chart generator tool

        Args:
            chart_type: Type of chart to generate
            data: Data for the chart
            title: Chart title
            x_label: Label for x-axis
            y_label: Label for y-axis
            colors: List of colors for chart elements
            width: Width of the chart in inches
            height: Height of the chart in inches
            dpi: Resolution of the chart

        Returns:
            Dictionary with base64-encoded chart image
        """
        try:
            # Create figure
            plt.figure(figsize=(width, height), dpi=dpi)

            # Generate chart based on type
            if chart_type == "bar":
                self._generate_bar_chart(data, title, x_label, y_label, colors)
            elif chart_type == "line":
                self._generate_line_chart(data, title, x_label, y_label, colors)
            elif chart_type == "pie":
                self._generate_pie_chart(data, title, colors)
            elif chart_type == "scatter":
                self._generate_scatter_chart(data, title, x_label, y_label, colors)
            elif chart_type == "histogram":
                self._generate_histogram(data, title, x_label, y_label, colors)
            else:
                return {"error": f"Unsupported chart type: {chart_type}"}

            # Save chart to bytes buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png")
            plt.close()

            # Convert to base64
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

            return {
                "chart_type": chart_type,
                "title": title,
                "image_data": f"data:image/png;base64,{image_base64}",
                "width": width,
                "height": height,
                "dpi": dpi,
            }
        except Exception as e:
            logger.error(f"Chart generation error: {e}")
            plt.close()  # Ensure figure is closed on error
            return {
                "error": f"Chart generation error: {str(e)}",
                "chart_type": chart_type,
            }

    def _generate_bar_chart(
        self,
        data: Dict[str, Any],
        title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        colors: Optional[List[str]] = None,
    ) -> None:
        """Generate a bar chart

        Args:
            data: Dictionary with 'labels' and 'values' keys
            title: Chart title
            x_label: Label for x-axis
            y_label: Label for y-axis
            colors: List of colors for bars
        """
        labels = data.get("labels", [])
        values = data.get("values", [])

        if not labels or not values:
            raise ValueError("Bar chart requires 'labels' and 'values' in data")

        if len(labels) != len(values):
            raise ValueError("Number of labels must match number of values")

        # Create bar chart
        bars = plt.bar(labels, values, color=colors)

        # Add data labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom",
            )

        # Set labels and title
        if title:
            plt.title(title)
        if x_label:
            plt.xlabel(x_label)
        if y_label:
            plt.ylabel(y_label)

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

    def _generate_line_chart(
        self,
        data: Dict[str, Any],
        title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        colors: Optional[List[str]] = None,
    ) -> None:
        """Generate a line chart

        Args:
            data: Dictionary with 'x' and 'y' keys, or 'series' for multiple lines
            title: Chart title
            x_label: Label for x-axis
            y_label: Label for y-axis
            colors: List of colors for lines
        """
        # Check if data contains multiple series
        if "series" in data:
            series_data = data["series"]
            if not isinstance(series_data, list):
                raise ValueError("Series data must be a list")

            # Plot each series
            for i, series in enumerate(series_data):
                label = series.get("label", f"Series {i+1}")
                x_values = series.get("x", [])
                y_values = series.get("y", [])

                if not x_values or not y_values:
                    raise ValueError(f"Series {i+1} requires 'x' and 'y' values")

                if len(x_values) != len(y_values):
                    raise ValueError(
                        f"Number of x values must match y values in series {i+1}"
                    )

                color = colors[i % len(colors)] if colors and len(colors) > 0 else None
                plt.plot(x_values, y_values, label=label, color=color)

            plt.legend()
        else:
            # Single line chart
            x_values = data.get("x", [])
            y_values = data.get("y", [])

            if not x_values or not y_values:
                raise ValueError("Line chart requires 'x' and 'y' values in data")

            if len(x_values) != len(y_values):
                raise ValueError("Number of x values must match number of y values")

            color = colors[0] if colors and len(colors) > 0 else None
            plt.plot(x_values, y_values, color=color)

        # Set labels and title
        if title:
            plt.title(title)
        if x_label:
            plt.xlabel(x_label)
        if y_label:
            plt.ylabel(y_label)

        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()

    def _generate_pie_chart(
        self,
        data: Dict[str, Any],
        title: Optional[str] = None,
        colors: Optional[List[str]] = None,
    ) -> None:
        """Generate a pie chart

        Args:
            data: Dictionary with 'labels' and 'values' keys
            title: Chart title
            colors: List of colors for pie slices
        """
        labels = data.get("labels", [])
        values = data.get("values", [])

        if not labels or not values:
            raise ValueError("Pie chart requires 'labels' and 'values' in data")

        if len(labels) != len(values):
            raise ValueError("Number of labels must match number of values")

        # Create pie chart
        plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
        plt.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle

        # Set title
        if title:
            plt.title(title)

        plt.tight_layout()

    def _generate_scatter_chart(
        self,
        data: Dict[str, Any],
        title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        colors: Optional[List[str]] = None,
    ) -> None:
        """Generate a scatter plot

        Args:
            data: Dictionary with 'x' and 'y' keys, or 'groups' for multiple series
            title: Chart title
            x_label: Label for x-axis
            y_label: Label for y-axis
            colors: List of colors for points
        """
        # Check if data contains multiple groups
        if "groups" in data:
            groups_data = data["groups"]
            if not isinstance(groups_data, list):
                raise ValueError("Groups data must be a list")

            # Plot each group
            for i, group in enumerate(groups_data):
                label = group.get("label", f"Group {i+1}")
                x_values = group.get("x", [])
                y_values = group.get("y", [])

                if not x_values or not y_values:
                    raise ValueError(f"Group {i+1} requires 'x' and 'y' values")

                if len(x_values) != len(y_values):
                    raise ValueError(
                        f"Number of x values must match y values in group {i+1}"
                    )

                color = colors[i % len(colors)] if colors and len(colors) > 0 else None
                plt.scatter(x_values, y_values, label=label, color=color)

            plt.legend()
        else:
            # Single scatter plot
            x_values = data.get("x", [])
            y_values = data.get("y", [])

            if not x_values or not y_values:
                raise ValueError("Scatter plot requires 'x' and 'y' values in data")

            if len(x_values) != len(y_values):
                raise ValueError("Number of x values must match number of y values")

            color = colors[0] if colors and len(colors) > 0 else None
            plt.scatter(x_values, y_values, color=color)

        # Set labels and title
        if title:
            plt.title(title)
        if x_label:
            plt.xlabel(x_label)
        if y_label:
            plt.ylabel(y_label)

        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()

    def _generate_histogram(
        self,
        data: Dict[str, Any],
        title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        colors: Optional[List[str]] = None,
    ) -> None:
        """Generate a histogram

        Args:
            data: Dictionary with 'values' key and optional 'bins' key
            title: Chart title
            x_label: Label for x-axis
            y_label: Label for y-axis
            colors: List of colors for histogram
        """
        values = data.get("values", [])
        bins = data.get("bins", 10)

        if not values:
            raise ValueError("Histogram requires 'values' in data")

        # Create histogram
        color = colors[0] if colors and len(colors) > 0 else None
        plt.hist(values, bins=bins, color=color, alpha=0.7, edgecolor="black")

        # Set labels and title
        if title:
            plt.title(title)
        if x_label:
            plt.xlabel(x_label)
        if y_label:
            plt.ylabel(y_label)

        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()


class HeatmapGeneratorTool(BaseTool):
    """Tool for generating heatmaps from matrix data"""

    def __init__(self):
        super().__init__(
            name="heatmap_generator", description="Generate heatmaps from matrix data"
        )
        self._parameters = {
            "data": {
                "type": "array",
                "description": "2D array/matrix of values for the heatmap",
            },
            "title": {"type": "string", "description": "Heatmap title"},
            "x_labels": {
                "type": "array",
                "description": "Labels for x-axis",
                "items": {"type": "string"},
            },
            "y_labels": {
                "type": "array",
                "description": "Labels for y-axis",
                "items": {"type": "string"},
            },
            "colormap": {
                "type": "string",
                "description": "Matplotlib colormap name",
                "default": "viridis",
            },
            "show_values": {
                "type": "boolean",
                "description": "Show values in heatmap cells",
                "default": True,
            },
            "width": {
                "type": "number",
                "description": "Width of the heatmap in inches",
                "default": 10,
            },
            "height": {
                "type": "number",
                "description": "Height of the heatmap in inches",
                "default": 8,
            },
            "dpi": {
                "type": "integer",
                "description": "Resolution of the heatmap",
                "default": 100,
            },
        }
        self._required_params = ["data"]
        logger.info("Initialized tool: heatmap_generator")

    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for tool parameters"""
        return {
            "type": "object",
            "properties": self._parameters,
            "required": self._required_params,
        }

    async def _execute(
        self,
        data: List[List[float]],
        title: Optional[str] = None,
        x_labels: Optional[List[str]] = None,
        y_labels: Optional[List[str]] = None,
        colormap: str = "viridis",
        show_values: bool = True,
        width: float = 10,
        height: float = 8,
        dpi: int = 100,
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute the heatmap generator tool

        Args:
            data: 2D array/matrix of values for the heatmap
            title: Heatmap title
            x_labels: Labels for x-axis
            y_labels: Labels for y-axis
            colormap: Matplotlib colormap name
            show_values: Show values in heatmap cells
            width: Width of the heatmap in inches
            height: Height of the heatmap in inches
            dpi: Resolution of the heatmap

        Returns:
            Dictionary with base64-encoded heatmap image
        """
        try:
            # Convert data to numpy array
            matrix = np.array(data)

            # Check if matrix is 2D
            if matrix.ndim != 2:
                raise ValueError("Heatmap requires a 2D matrix")

            # Create figure
            plt.figure(figsize=(width, height), dpi=dpi)

            # Generate heatmap
            im = plt.imshow(matrix, cmap=colormap)

            # Add colorbar
            plt.colorbar(im)

            # Set labels
            if x_labels:
                if len(x_labels) != matrix.shape[1]:
                    raise ValueError("Number of x labels must match matrix width")
                plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha="right")

            if y_labels:
                if len(y_labels) != matrix.shape[0]:
                    raise ValueError("Number of y labels must match matrix height")
                plt.yticks(range(len(y_labels)), y_labels)

            # Set title
            if title:
                plt.title(title)

            # Add values to cells
            if show_values:
                for i in range(matrix.shape[0]):
                    for j in range(matrix.shape[1]):
                        text = plt.text(
                            j,
                            i,
                            f"{matrix[i, j]:.2f}",
                            ha="center",
                            va="center",
                            color="w" if matrix[i, j] > np.mean(matrix) else "black",
                        )

            plt.tight_layout()

            # Save heatmap to bytes buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png")
            plt.close()

            # Convert to base64
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

            return {
                "title": title,
                "image_data": f"data:image/png;base64,{image_base64}",
                "width": width,
                "height": height,
                "dpi": dpi,
                "matrix_shape": matrix.shape,
            }
        except Exception as e:
            logger.error(f"Heatmap generation error: {e}")
            plt.close()  # Ensure figure is closed on error
            return {"error": f"Heatmap generation error: {str(e)}"}
