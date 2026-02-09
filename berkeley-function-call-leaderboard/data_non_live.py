import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    return (pl,)


@app.cell
def _(pl):
    pl.read_csv("score/data_non_live.csv")
    return


if __name__ == "__main__":
    app.run()
