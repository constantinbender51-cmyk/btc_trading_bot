# railway_entry.py  (chooses worker vs web)
import os

if os.getenv("RUN_MODE") == "web":
    import uvicorn
    from railway_app import app
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
else:
    import main
    main.main()
