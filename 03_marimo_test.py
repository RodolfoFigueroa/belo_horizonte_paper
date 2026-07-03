# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.3",
# ]
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import math

    return (math,)


@app.cell
def _(math):
    math.log(1)
    return


@app.cell
def _():
    1
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
