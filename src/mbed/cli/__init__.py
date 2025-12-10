from .cli import app
from .commands import init, search, status, update


app.add_typer(init)
app.add_typer(search)
app.add_typer(status)
app.add_typer(update)


__all__ = ["app"]

