import time

from airflow.providers.http.hooks.http import HttpHook

CONN_ID = "sismosapi"
EVENT_QUERY = "/fdsnws/event/1/query"


def fetch(
    starttime,
    endtime,
    format="csv",
    eventtype="earthquake",
    minmagnitude=0,
    timeout=30,
    retries=3,
):
    hook = HttpHook(method="GET", http_conn_id=CONN_ID)
    for intento in range(retries):
        try:
            resp = hook.run(
                endpoint=EVENT_QUERY,
                data={
                    "format": format,
                    "starttime": starttime,
                    "endtime": endtime,
                    "eventtype": eventtype,
                    "minmagnitude": minmagnitude,
                },
                extra_options={"check_response": True},
            )

            return resp
        except Exception:
            if intento == retries - 1:
                raise
            time.sleep(2 * (intento + 1))
