import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.security import hash_password
from app.db.session import SessionLocal
from app.models.user import User


def seed_admin(update_password: bool = False) -> None:
    admin_email = os.getenv("ADMIN_EMAIL", "admin@example.com")
    admin_full_name = os.getenv("ADMIN_FULL_NAME", "Main Admin")
    admin_password = os.getenv("ADMIN_PASSWORD", "admin1234")

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == admin_email).first()

        if user is None:
            user = User(
                email=admin_email,
                full_name=admin_full_name,
                hashed_password=hash_password(admin_password),
                role="admin",
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            print(f"Created admin user: {user.email}")
            return

        changed = False

        if user.role != "admin":
            user.role = "admin"
            changed = True

        if update_password:
            user.hashed_password = hash_password(admin_password)
            changed = True

        if changed:
            db.commit()
            db.refresh(user)
            print(f"Updated admin user: {user.email}")
        else:
            print(f"Admin user already present: {user.email}")
    finally:
        db.close()


if __name__ == "__main__":
    force_password_update = os.getenv("ADMIN_FORCE_PASSWORD_UPDATE", "false").lower() == "true"
    seed_admin(update_password=force_password_update)
