# Source - https://stackoverflow.com/a/73704579
# Posted by isle_of_gods, modified by community. See post 'Timeline' for change history
# Retrieved 2026-02-05, License - CC BY-SA 4.0


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        """Track validation loss trends for early stopping."""
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        """Return True when validation loss has plateaued long enough."""
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
