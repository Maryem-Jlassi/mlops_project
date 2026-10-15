"""Prefect deployment - daily scheduling of flows (Prefect v3)."""

from pipeline_prefect import all_flow, train_flow, evaluate_flow, code_flow

if __name__ == "__main__":
    # flow.serve() creates a scheduled deployment and runs a long-lived
    # process that listens for scheduled runs.
    # Each serve() call blocks, so run ONE flow per terminal if you want
    # all of them scheduled at the same time.

    # Flow "code" : every day at 1:00 AM
    code_flow.serve(name="ml-pipeline-code", cron="0 1 * * *")

    # Flow "all" : every day at 2:00 AM
    all_flow.serve(name="ml-pipeline-all", cron="0 2 * * *")

    # Flow "train" : every day at 3:00 AM
    train_flow.serve(name="ml-pipeline-train", cron="0 3 * * *")

    # Flow "evaluate" : every day at 4:00 AM
    evaluate_flow.serve(name="ml-pipeline-evaluate", cron="0 4 * * *")
