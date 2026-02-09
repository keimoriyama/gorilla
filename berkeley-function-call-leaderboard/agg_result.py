import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import polars as pl
    return json, pl


@app.cell
def _(json, pl):
    def load_data(path: str):
        with open(path) as f:
            data = [json.loads(fi) for fi in f.readlines()[1:]]
        return data

    def agg_err(path: str):
        data = load_data(path)
        return (
            pl.DataFrame(data, strict=False)
            .group_by("error_type")
            .agg(pl.len())
            .sort("len", descending=True)
        )
    return agg_err, load_data


@app.cell
def _(agg_err):
    agg_err("score/llm-jp-3.1-13b-instruct4/non_live/BFCL_v4_simple_python_score.json")
    return


@app.cell
def _(agg_err):
    agg_err("score/llm-jp-3.1-13b-instruct4-xlam/non_live/BFCL_v4_simple_python_score.json")
    return


@app.cell
def _(agg_err):
    agg_err("score/llm-jp-3-13b-instruct3/non_live/BFCL_v4_simple_python_score.json")
    return


@app.cell
def _():
    272 / (272 + 30)
    return


@app.cell
def _(load_data):
    _data = load_data(
        "score/llm-jp-3.1-13b-instruct4/non_live/BFCL_v4_simple_python_score.json"
    )
    _data
    return


@app.cell
def _(load_data):
    load_data(
        "score/llm-jp-3-13b-instruct3/non_live/BFCL_v4_simple_python_score.json"
    )
    return


if __name__ == "__main__":
    app.run()
