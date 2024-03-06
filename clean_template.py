import re
from pathlib import Path
from subprocess import check_output


if __name__ == "__main__":

    # Get the name of the repo and parse org and bench name
    repo_url = check_output(['git', 'remote', 'get-url', 'origin']).decode()
    name = re.split('github.com[:/]', repo_url)[1].strip()
    ORG, BENCHMARK_NAME = name.replace('.git', '').split('/')

    # update readme to have the right badge
    README = Path('README.rst')
    text = README.read_text()

    text = text.replace('#ORG', ORG)
    text = text.replace('#BENCHMARK_NAME', BENCHMARK_NAME)

    text = '\n'.join([line for line in text.splitlines()[13:]
                      if 'template_benchmark' not in line] + [''])
    README.write_text(text)

    # update the repo URL in the objective.py file
    file = Path("objective.py")
    text = file.read_text()
    text = text.replace('#ORG', ORG)
    text = text.replace('#BENCHMARK_NAME', BENCHMARK_NAME)
    file.write_text(text)

    # Remove files specific to the template repo
    to_remove = [
        Path(".github") / "workflows" / "test_benchmarks.yml",
        Path(".github") / "workflows" / "lint_benchmarks.yml",
        Path(".") / "clean_template.py"
    ]
    for file in to_remove:
        if file.exists():
            file.unlink()
