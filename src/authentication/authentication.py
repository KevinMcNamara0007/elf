from fastapi import HTTPException

from src.utilities.general import API_TOKENS, NO_TOKEN


def verify_token(usr_token):
    if usr_token == NO_TOKEN:
        raise HTTPException(status_code=401, detail="Token is missing.")
    for sys_token in API_TOKENS:
        if usr_token == sys_token:
            return True
    print(usr_token)
    raise HTTPException(status_code=401, detail=f"Unauthorized Token: {usr_token}")
