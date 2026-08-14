from __future__ import annotations

from typing import Any


class WandbLogger:
    def __init__(self, *, enabled: bool, project: str, group: str, name: str, config: dict[str, Any]):
        self.run = None
        if enabled:
            import wandb

            self.run = wandb.init(project=project, group=group, name=name, config=config, resume="allow")

    @property
    def id(self) -> str | None:
        return self.run.id if self.run is not None else None

    def log(self, values: dict[str, Any], step: int | None = None) -> None:
        if self.run is not None:
            self.run.log(values, step=step)

    def table(self, name: str, columns: list[str], rows: list[list[Any]], step: int | None = None) -> None:
        if self.run is not None:
            import wandb

            self.run.log({name: wandb.Table(columns=columns, data=rows)}, step=step)

    def save(self, path: str) -> None:
        if self.run is not None:
            self.run.save(path)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
