import fire

from src import __version__


class PansharpeningCli:
    @property
    def version(self) -> str:
        return __version__


def main() -> None:
    fire.Fire(PansharpeningCli)


if __name__ == "__main__":
    main()