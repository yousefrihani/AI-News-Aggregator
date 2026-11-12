"""Allow ``python -m Articles`` to run the command-line interface."""

from Articles.cli import main

raise SystemExit(main())
