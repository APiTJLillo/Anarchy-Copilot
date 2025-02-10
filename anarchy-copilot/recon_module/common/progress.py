"""Progress tracking for reconnaissance operations."""

class ReconProgress:
    def __init__(self):
        self.current = 0
        self.total = 100
        self.message = ""

    def update(self, current: int, total: int, message: str):
        """Update progress status."""
        self.current = current
        self.total = total
        self.message = message

    def get_status(self) -> dict:
        """Get current progress status."""
        return {
            "current": self.current,
            "total": self.total,
            "message": self.message
        }
