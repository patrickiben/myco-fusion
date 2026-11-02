"""Rebuild everything from raw data: python -m study (needs pip install ".[study]")."""
from . import build_report, coverage, make_assets, phone_power, run_study

make_assets.main()
run_study.main()
coverage.main()
phone_power.main()
build_report.main()
