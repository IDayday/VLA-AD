"""Frozen native evaluator, with a narrowly authorized A diagnostic admission."""
import sys
from correlation_runtime import CODE, evaluator_source, verify_scope

if __name__ == '__main__':
    verify_scope()
    path = CODE / 'student_distillation/evaluate.py'
    arguments = sys.argv[1:]
    if arguments and arguments[0] == '--':
        arguments = arguments[1:]
    sys.argv = ['student_distillation.evaluate', *arguments]
    namespace = {'__name__': '__main__', '__package__': 'student_distillation',
                 '__file__': str(path), '__builtins__': __builtins__}
    exec(compile(evaluator_source(path.read_text()), str(path) + ':A_correlation', 'exec'), namespace)
