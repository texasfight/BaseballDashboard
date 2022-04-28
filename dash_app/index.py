from app import app, server

if __name__ == "__main__":
    app.run_server(
        host="0.0.0.0",
        port=8085,
        debug=False,
        dev_tools_props_check=False
    )
