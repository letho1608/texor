import os
import sys
import warnings
import time

# Thiết lập môi trường
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONWARNINGS'] = 'ignore'

# Import các module cần thiết
import click
from rich.console import Console
from rich.panel import Panel
import platform
from ..version import __version__

# Thiết lập console
console = Console()

def show_header():
    """Display Texor header"""
    console.print(Panel(
        "[bold blue]Texor[/bold blue] - Native Deep Learning Framework\n" +
        "[dim]Lightweight ML library with PyTorch-style API[/dim]",
        subtitle=f"v{__version__}",
        style="green"
    ))

@click.group()
@click.version_option(version=__version__)
def cli():
    """Texor CLI - Native deep learning framework"""
    show_header()

@cli.command()
def info():
    """Display information about environment and installation"""
    # Basic information
    system_info = [
        ("Python", platform.python_version()),
        ("Platform", platform.platform()),
        ("Texor Version", __version__),
    ]
    
    # Available components (accurate list)
    modules = {
        "Core": ["Tensor", "Autograd", "Device Management", "Native Backend"],
        "Neural Networks": ["Linear", "Conv2D", "MaxPool2D", "BatchNorm2D", "Sequential"],
        "Activations": ["ReLU", "Sigmoid", "Tanh"],
        "Loss Functions": ["MSELoss", "CrossEntropyLoss", "BCELoss", "L1Loss", "HuberLoss", "SmoothL1Loss", "KLDivLoss"],
        "Optimizers": ["SGD", "Adam", "RMSprop", "Adagrad", "Adadelta"],
        "Data": ["Dataset", "DataLoader", "Transform", "Compose", "Normalize", "ToTensor"],
        "Models": ["Model"]
    }
    
    # Display system information
    console.print("\n[bold]System Information:[/bold]")
    for key, value in system_info:
        console.print(f"[yellow]{key}:[/yellow] {value}")
    
    # Display available modules
    console.print("\n[bold]Available Modules:[/bold]")
    for category, items in modules.items():
        console.print(f"\n[cyan]{category}[/cyan]")
        for item in items:
            console.print(f"  [green]•[/green] {item}")
        console.print()

@cli.command()
@click.argument('package_name', required=False)
def list(package_name=None):
    """List installed modules"""
    categories = {
        "Deep Learning": [
            "Dense", "CNN", "RNN", "Transformer",
            "Attention", "AutoEncoder"
        ],
        "Machine Learning": [
            "LinearRegression", "LogisticRegression",
            "RandomForest", "GradientBoosting",
            "KMeans", "DBSCAN"
        ],
        "AutoML": [
            "NAS", "HyperTuner",
            "ModelSelector", "AutoEnsemble"
        ],
        "Utils": [
            "DataLoader", "Preprocessor",
            "Visualizer", "Metrics"
        ]
    }

    if package_name:
        console.print(f"\n[bold]Tìm kiếm cho '{package_name}':[/bold]")
        found = False
        for category, modules in categories.items():
            matched = [m for m in modules if package_name.lower() in m.lower()]
            if matched:
                found = True
                console.print(f"\n[cyan]{category}[/cyan]")
                for module in matched:
                    console.print(f"  [green]•[/green] {module}")
                console.print()
        if not found:
            console.print("[red]Không tìm thấy module phù hợp![/red]")
    else:
        for category, modules in categories.items():
            console.print(f"\n[cyan]{category}[/cyan]")
            for module in modules:
                console.print(f"  [green]•[/green] {module}")
            console.print()

@cli.command()
def check():
    """Kiểm tra môi trường và dependencies"""
    console.print("\n[bold]Đang kiểm tra môi trường...[/bold]")
    
    # Kiểm tra Python version
    python_ok = sys.version_info >= (3, 8)
    
    # Kiểm tra dependencies cho native library
    try:
        import numpy as np
        import numba
        deps_ok = True
        console.print("[green]✓[/green] NumPy và Numba đã sẵn sàng")
        
        # Kiểm tra GPU support (optional)
        try:
            import cupy as cp
            console.print("[green]✓[/green] CuPy có sẵn - hỗ trợ GPU")
        except ImportError:
            console.print("[yellow]⚠[/yellow] CuPy không có - chỉ sử dụng CPU")
            
    except ImportError as e:
        deps_ok = False
        console.print(f"[red]✗[/red] Thiếu dependency: {e}")
            
    if python_ok and deps_ok:
        console.print("\n[green]✓[/green] Tất cả kiểm tra đều thành công!")
        console.print("[dim]Texor native library sẵn sàng sử dụng![/dim]")
    else:
        console.print("\n[red]✗[/red] Phát hiện một số vấn đề:")
        if not python_ok:
            console.print("  [red]•[/red] Yêu cầu Python 3.8 trở lên")
        if not deps_ok:
            console.print("  [red]•[/red] Thiếu một số dependencies quan trọng")

def main():
    cli()

if __name__ == '__main__':
    main()