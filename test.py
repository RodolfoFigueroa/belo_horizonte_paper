import marimo._code_mode as cm

async with cm.get_context(skip_validation=True, skip_staleness_check=True) as ctx:
    cid = ctx.create_cell("print('asdasdasd')", hide_code=False)
    ctx.run_cell(cid)
