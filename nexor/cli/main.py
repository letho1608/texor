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
    """Hiển thị header của Nexor với màu sắc"""
    console.print(Panel(
        "[bold blue]Nexor[/bold blue] - Framework AI Toàn Diện\n" +
        "[dim]Kết hợp Deep Learning, Machine Learning và AutoML[/dim]",
        subtitle=f"v{__version__}",
        style="green"
    ))

@click.group()
@click.version_option(version=__version__)
def cli():
    """Nexor CLI - Framework AI toàn diện cho cộng đồng AI Việt Nam"""
    show_header()

@cli.command()
def info():
    """Hiển thị thông tin về môi trường và cài đặt"""
    # Thông tin cơ bản
    system_info = [
        ("Python", platform.python_version()),
        ("Platform", platform.platform()),
        ("Nexor Version", __version__),
    ]
    
    # Thông tin về các module
    modules = {
        "Deep Learning": ["Neural Networks", "CNN", "RNN", "Transformer"],
        "Machine Learning": ["Classification", "Regression", "Clustering"],
        "AutoML": ["Architecture Search", "Hyperparameter Tuning"],
        "Data Processing": ["Preprocessing", "Augmentation", "Visualization"]
    }
    
    # Hiển thị thông tin hệ thống
    console.print("\n[bold]Thông tin hệ thống:[/bold]")
    for key, value in system_info:
        console.print(f"[yellow]{key}:[/yellow] {value}")
    
    # Hiển thị thông tin module
    console.print("\n[bold]Các module có sẵn:[/bold]")
    for category, items in modules.items():
        console.print(f"\n[cyan]{category}[/cyan]")
        for item in items:
            console.print(f"  [green]•[/green] {item}")
        console.print()

@cli.command()
@click.argument('package_name', required=False)
def list(package_name=None):
    """Liệt kê các module đã cài đặt"""
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
    
    # Kiểm tra dependencies
    try:
        import tensorflow as tf
        import torch
        import numpy as np
        import sklearn
        deps_ok = True
    except ImportError:
        deps_ok = False
            
    if python_ok and deps_ok:
        console.print("[green]✓[/green] Tất cả kiểm tra đều thành công!")
    else:
        console.print("[red]✗[/red] Phát hiện một số vấn đề:")
        if not python_ok:
            console.print("  [red]•[/red] Yêu cầu Python 3.8 trở lên")
        if not deps_ok:
            console.print("  [red]•[/red] Thiếu một số dependencies quan trọng")

def main():
    cli()

if __name__ == '__main__':
    main()