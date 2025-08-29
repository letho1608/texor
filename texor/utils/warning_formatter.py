import logging
import sys
from rich.console import Console
from rich.panel import Panel

console = Console()

class TexorWarningFormatter(logging.Handler):
    """Handler tùy chỉnh để định dạng các cảnh báo theo style của Texor"""
    
    def __init__(self):
        super().__init__()
        self.setLevel(logging.INFO)
        
    def emit(self, record):
        """Định dạng và hiển thị cảnh báo với style của Texor"""
        if "tensorflow" in record.name.lower():
            # Định dạng cảnh báo TensorFlow
            msg = record.getMessage()
            if "oneDNN" in msg:
                # Bỏ qua cảnh báo oneDNN
                return
                
            console.print(Panel(
                f"[yellow]{msg}[/yellow]",
                title="[bold yellow]TensorFlow Warning[/bold yellow]",
                border_style="yellow"
            ))
        else:
            # Định dạng các cảnh báo khác
            console.print(Panel(
                f"[yellow]{record.getMessage()}[/yellow]",
                title=f"[bold yellow]{record.name} Warning[/bold yellow]",
                border_style="yellow"
            ))

def setup_warning_handler():
    """Thiết lập handler để xử lý tất cả các cảnh báo"""
    # Xóa tất cả các handler mặc định
    root = logging.getLogger()
    root.handlers = []
    
            # Thêm handler tùy chỉnh của Texor
        handler = TexorWarningFormatter()
    root.addHandler(handler)
    
    # Cập nhật TensorFlow logger
    tf_logger = logging.getLogger('tensorflow')
    tf_logger.handlers = []
    tf_logger.addHandler(handler)
    
    # Điều hướng stderr để bắt các cảnh báo từ C++
    class StderrHandler:
        def write(self, text):
            if text.strip():
                if "tensorflow" in text.lower():
                    console.print(Panel(
                        f"[yellow]{text.strip()}[/yellow]",
                        title="[bold yellow]TensorFlow System[/bold yellow]",
                        border_style="yellow"
                    ))
                else:
                    sys.__stderr__.write(text)
        
        def flush(self):
            sys.__stderr__.flush()
    
    sys.stderr = StderrHandler()