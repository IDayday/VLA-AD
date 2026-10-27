"""Run the frozen numerical evaluator with the explicit new Test admission."""
import sys
from test_pdms_runtime import CODE, evaluator_source, verify_scope


def main():
    verify_scope()
    path = CODE / 'student_distillation/evaluate.py'
    arguments = sys.argv[1:]
    if arguments and arguments[0] == '--':
        arguments = arguments[1:]
    sys.argv = ['student_distillation.evaluate', *arguments]
    namespace = {'__name__': '__main__', '__package__': 'student_distillation',
                 '__file__': str(path), '__builtins__': __builtins__}
    exec(compile(evaluator_source(path.read_text()), str(path) + ':test_selection', 'exec'), namespace)


if __name__ == '__main__':
    main()
