import json
import os
from rich.table import Table
from rich.console import Console, ConsoleOptions

from reporter.Report import Report


class Reporter:
    '''Generate statistics report.'''

    __text_path: str
    __json_path: str

    def __init__(
        self,
        text_path: str = 'report.txt',
        json_path: str = 'report.json',
    ) -> None:
        self.set_text_path(text_path)
        self.set_json_path(json_path)

    @property
    def text_path(self) -> str:
        return self.__text_path

    @text_path.setter
    def text_path(self, path: str) -> None:
        self.set_text_path(path)

    def set_text_path(self, path: str) -> None:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        self.__text_path = path

    @property
    def json_path(self) -> str:
        return self.__json_path

    def set_json_path(self, path: str) -> None:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        self.__json_path = path

    @json_path.setter
    def json_path(self, path: str) -> None:
        self.set_json_path(path)

    def __call__(self, report: Report) -> str:
        return self.to_text(report)

    def append_json(self, report: Report) -> None:
        if not os.path.exists(self.__json_path):
            with open(self.__json_path, 'w', encoding='utf-8') as f:
                json.dump([], f)
        with open(self.__json_path, 'r', encoding='utf-8') as f:
            report_json_arr = json.load(f)
        report_json_arr.append(report)
        with open(self.__json_path, 'w', encoding='utf-8') as f:
            json.dump(report_json_arr, f, ensure_ascii=False, indent=2)

    def write_json(self, report: Report) -> None:
        with open(self.__json_path, 'w', encoding='utf-8') as f:
            json.dump([report], f, ensure_ascii=False, indent=2)

    def clear_json(self) -> None:
        with open(self.__json_path, 'w', encoding='utf-8') as f:
            json.dump([], f)

    def print(self, report: Report) -> None:
        console = Console()
        tables = self.to_tables(report)
        for table in tables:
            console.print(table)
            console.print()

    def write(self, report: Report) -> None:
        report_text = self.to_text(report)
        with open(self.__text_path, 'w', encoding='utf-8') as f:
            f.write(report_text + '\n')

    def append(self, report: Report) -> None:
        report_text = self.to_text(report)
        with open(self.__text_path, 'a', encoding='utf-8') as f:
            f.write(report_text + '\n')

    def clear(self) -> None:
        with open(self.__text_path, 'w', encoding='utf-8') as f:
            f.write('')

    def to_text(self, report: Report) -> str:
        table = self.to_tables(report)
        console = Console()
        return '\n'.join(''.join(line.text for line in console.render(t)) for t in table)

    def to_tables(self, report: Report) -> 'list[Table]':
        table = self.__make_blank_table(report, 'Basic Information')
        table.add_row('Generated time', report['time'])
        table.add_row('Cost time', f'{report["cost_time"]:.2f} seconds')
        tables = [table]
        table = self.__make_blank_table(report, 'Character Level Report')
        table.add_row('Correct', str(report['char_level']['correct']))
        table.add_row('Precision count', str(report['char_level']['precision_count']))
        table.add_row('Recall count', str(report['char_level']['recall_count']))
        table.add_row('Precision', f'{report["char_level"]["precision"]:.2%}')
        table.add_row('Recall', f'{report["char_level"]["recall"]:.2%}')
        table.add_row('F1', f'{report["char_level"]["f1"]:.4f}')
        tables.append(table)

        table = self.__make_blank_table(report, 'Sentence Level Report')
        table.add_row('Sentence count', str(report['sentence_level']['sentence_count']))
        table.add_row('At least one correct', f'{report["sentence_level"]["at_least_one_correct"]:.2%}')
        table.add_row('At least one correct count', str(report['sentence_level']['at_least_one_correct_count']))
        table.add_row('Totally correct', f'{report["sentence_level"]["totally_correct"]:.2%}')
        table.add_row('Totally correct count', str(report['sentence_level']['totally_correct_count']))
        tables.append(table)

        return tables

    # TODO Rename this here and in `to_tables`
    def __make_blank_table(self, report, subtitle: str):
        result = Table(title=f'{subtitle} ({report["recognizer"]})')
        result.add_column(
            subtitle,
            justify='left',
            style='cyan',
            no_wrap=True,
            width=30,
            header_style='bright_cyan',
        )
        result.add_column(
            'Value',
            justify='right',
            style='magenta',
            no_wrap=True,
            width=20,
            header_style='bright_magenta',
        )
        return result
