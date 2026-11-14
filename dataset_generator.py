# dataset_generator.py
"""
Synthetic e-commerce dataset generator.

Generates:
- users.json
- products.json
- categories.json
- transactions.json
- sessions_0.json, sessions_1.json, ...

Install dependencies:
    pip install numpy Faker

Run:
    python dataset_generator.py
"""

import datetime
import json
import random
import threading
import uuid

import numpy as np
from faker import Faker


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NUM_USERS = 10_000
NUM_PRODUCTS = 5_000
NUM_CATEGORIES = 25
NUM_TRANSACTIONS = 500_000
NUM_SESSIONS = 2_000_000
TIMESPAN_DAYS = 90
MAX_ITERATIONS = (NUM_SESSIONS + NUM_TRANSACTIONS) * 2
CHUNK_SIZE = 100_000

# Reproducible random data
np.random.seed(42)
random.seed(42)
Faker.seed(42)

fake = Faker()


# ---------------------------------------------------------------------------
# ID generators
# ---------------------------------------------------------------------------

def generate_session_id():
    """Return a unique session identifier."""
    return f"sess_{uuid.uuid4().hex[:10]}"


def generate_transaction_id():
    """Return a unique transaction identifier."""
    return f"txn_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Inventory management
# ---------------------------------------------------------------------------

class InventoryManager:
    """Manage product stock safely when inventory is updated."""

    def __init__(self, products):
        self.products = {product["product_id"]: product for product in products}
        self.lock = threading.RLock()

    def update_stock(self, product_id, quantity):
        """Reduce stock if the product exists and enough stock is available."""
        with self.lock:
            if product_id not in self.products:
                return False

            product = self.products[product_id]
            if product["current_stock"] >= quantity:
                product["current_stock"] -= quantity
                return True

            return False

    def get_product(self, product_id):
        """Return a product by ID."""
        with self.lock:
            return self.products.get(product_id)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def determine_page_type(position, previous_pages):
    """Choose a page type based on the visitor's previous page."""
    if position == 0:
        return random.choice(["home", "search", "category_listing"])

    if not previous_pages:
        return "home"

    previous_page = previous_pages[-1]["page_type"]

    transitions = {
        "home": (
            ["category_listing", "search", "product_detail"],
            [0.5, 0.3, 0.2],
        ),
        "category_listing": (
            ["product_detail", "category_listing", "search", "home"],
            [0.7, 0.1, 0.1, 0.1],
        ),
        "search": (
            ["product_detail", "search", "category_listing", "home"],
            [0.6, 0.2, 0.1, 0.1],
        ),
        "product_detail": (
            ["product_detail", "cart", "category_listing", "search", "home"],
            [0.3, 0.3, 0.2, 0.1, 0.1],
        ),
        "cart": (
            ["checkout", "product_detail", "category_listing", "home"],
            [0.6, 0.2, 0.1, 0.1],
        ),
        "checkout": (
            ["confirmation", "cart", "home"],
            [0.8, 0.1, 0.1],
        ),
        "confirmation": (
            ["home", "product_detail", "category_listing"],
            [0.6, 0.2, 0.2],
        ),
    }

    if previous_page not in transitions:
        return "home"

    page_types, weights = transitions[previous_page]
    return random.choices(page_types, weights=weights)[0]


def get_page_content(page_type, products_list, categories_list, inventory):
    """Return a suitable product/category for the requested page type."""
    if page_type == "product_detail":
        for _ in range(10):
            product = random.choice(products_list)
            if product["is_active"] and product["current_stock"] > 0:
                category_id = product["category_id"]
                category = next(
                    (
                        item
                        for item in categories_list
                        if item["category_id"] == category_id
                    ),
                    None,
                )
                return product, category

        product = random.choice(products_list)
        category = next(
            (
                item
                for item in categories_list
                if item["category_id"] == product["category_id"]
            ),
            None,
        )
        return product, category

    if page_type == "category_listing":
        return None, random.choice(categories_list)

    return None, None


def json_serializer(obj):
    """Serialize date and datetime objects as ISO-formatted strings."""
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()

    raise TypeError(f"Type {type(obj)} is not serializable")


# ---------------------------------------------------------------------------
# Category generation
# ---------------------------------------------------------------------------

def generate_categories():
    """Generate product categories and their subcategories."""
    categories = []

    for category_id in range(NUM_CATEGORIES):
        category = {
            "category_id": f"cat_{category_id:03d}",
            "name": fake.company(),
            "subcategories": [],
        }

        for subcategory_id in range(random.randint(3, 5)):
            category["subcategories"].append(
                {
                    "subcategory_id": (
                        f"sub_{category_id:03d}_{subcategory_id:02d}"
                    ),
                    "name": fake.bs(),
                    "profit_margin": round(random.uniform(0.1, 0.4), 2),
                }
            )

        categories.append(category)

    print(f"Generated {len(categories):,} categories")
    return categories


# ---------------------------------------------------------------------------
# Product generation
# ---------------------------------------------------------------------------

def generate_products(categories):
    """Generate products, including price history and initial stock."""
    products = []
    product_creation_start = (
        datetime.datetime.now()
        - datetime.timedelta(days=TIMESPAN_DAYS * 2)
    )

    for product_id in range(NUM_PRODUCTS):
        category = random.choice(categories)
        base_price = round(random.uniform(5, 500), 2)
        price_history = []

        initial_date = fake.date_time_between(
            start_date=product_creation_start,
            end_date=(
                product_creation_start
                + datetime.timedelta(days=TIMESPAN_DAYS // 3)
            ),
        )

        price_history.append(
            {
                "price": base_price,
                "date": initial_date.isoformat(),
            }
        )

        for _ in range(random.randint(0, 2)):
            price_change_date = fake.date_time_between(
                start_date=initial_date,
                end_date="now",
            )
            new_price = round(base_price * random.uniform(0.8, 1.2), 2)

            price_history.append(
                {
                    "price": new_price,
                    "date": price_change_date.isoformat(),
                }
            )
            initial_date = price_change_date

        price_history.sort(key=lambda item: item["date"])
        current_price = price_history[-1]["price"]

        products.append(
            {
                "product_id": f"prod_{product_id:05d}",
                "name": fake.catch_phrase().title(),
                "category_id": category["category_id"],
                "base_price": current_price,
                "current_stock": random.randint(10, 1000),
                "is_active": random.choices(
                    [True, False],
                    weights=[0.95, 0.05],
                )[0],
                "price_history": price_history,
                "creation_date": price_history[0]["date"],
            }
        )

    print(f"Generated {len(products):,} products")
    return products


# ---------------------------------------------------------------------------
# User generation
# ---------------------------------------------------------------------------

def generate_users():
    """Generate user profiles and registration/activity dates."""
    users = []

    for user_id in range(NUM_USERS):
        registration_date = fake.date_time_between(
            start_date=f"-{TIMESPAN_DAYS * 3}d",
            end_date=f"-{TIMESPAN_DAYS}d",
        )

        users.append(
            {
                "user_id": f"user_{user_id:06d}",
                "geo_data": {
                    "city": fake.city(),
                    "state": fake.state_abbr(),
                    "country": fake.country_code(),
                },
                "registration_date": registration_date.isoformat(),
                "last_active": fake.date_time_between(
                    start_date=registration_date,
                    end_date="now",
                ).isoformat(),
            }
        )

    print(f"Generated {len(users):,} users")
    return users


# ---------------------------------------------------------------------------
# Session and transaction generation
# ---------------------------------------------------------------------------

def generate_sessions_and_transactions(users, products, categories, inventory):
    """Generate browsing sessions and transactions."""
    sessions = []
    transactions = []
    transaction_counter = 0
    session_counter = 0
    iteration = 0

    print("Generating sessions and transactions...")

    while (
        session_counter < NUM_SESSIONS
        or transaction_counter < NUM_TRANSACTIONS
    ) and iteration < MAX_ITERATIONS:
        iteration += 1

        # Generate one browsing session.
        if session_counter < NUM_SESSIONS:
            user = random.choice(users)
            session_id = generate_session_id()
            session_start = fake.date_time_between(
                start_date=f"-{TIMESPAN_DAYS}d",
                end_date="now",
            )
            session_duration = random.randint(30, 3600)

            page_views = []
            viewed_products = set()
            cart_contents = {}

            time_slots = sorted(
                [0]
                + [
                    random.randint(1, session_duration - 1)
                    for _ in range(random.randint(3, 15))
                ]
                + [session_duration]
            )

            for position in range(len(time_slots) - 1):
                view_duration = time_slots[position + 1] - time_slots[position]
                page_type = determine_page_type(position, page_views)
                product, category = get_page_content(
                    page_type,
                    products,
                    categories,
                    inventory,
                )

                if page_type == "product_detail" and product:
                    product_id = product["product_id"]
                    viewed_products.add(product_id)

                    if random.random() < 0.3:
                        if product_id not in cart_contents:
                            cart_contents[product_id] = {
                                "quantity": 0,
                                "price": product["base_price"],
                            }

                        current_product = inventory.get_product(product_id)
                        max_possible = min(
                            3,
                            current_product["current_stock"]
                            - cart_contents[product_id]["quantity"],
                        )

                        if max_possible > 0:
                            added_quantity = random.randint(1, max_possible)
                            cart_contents[product_id]["quantity"] += added_quantity

                page_views.append(
                    {
                        "timestamp": (
                            session_start
                            + datetime.timedelta(seconds=time_slots[position])
                        ).isoformat(),
                        "page_type": page_type,
                        "product_id": product["product_id"] if product else None,
                        "category_id": (
                            category["category_id"] if category else None
                        ),
                        "view_duration": view_duration,
                    }
                )

            converted = False
            reached_checkout = any(
                page["page_type"] in ["checkout", "confirmation"]
                for page in page_views
            )

            if cart_contents and reached_checkout:
                converted = random.random() < 0.7

            session_geo = user["geo_data"].copy()
            session_geo["ip_address"] = fake.ipv4()

            sessions.append(
                {
                    "session_id": session_id,
                    "user_id": user["user_id"],
                    "start_time": session_start.isoformat(),
                    "end_time": (
                        session_start
                        + datetime.timedelta(seconds=session_duration)
                    ).isoformat(),
                    "duration_seconds": session_duration,
                    "geo_data": session_geo,
                    "device_profile": {
                        "type": random.choice(["mobile", "desktop", "tablet"]),
                        "os": random.choice(
                            ["iOS", "Android", "Windows", "macOS"]
                        ),
                        "browser": random.choice(
                            ["Chrome", "Safari", "Firefox", "Edge"]
                        ),
                    },
                    "viewed_products": list(viewed_products),
                    "page_views": page_views,
                    "cart_contents": {
                        key: value
                        for key, value in cart_contents.items()
                        if value["quantity"] > 0
                    },
                    "conversion_status": (
                        "converted"
                        if converted
                        else "abandoned"
                        if cart_contents
                        else "browsed"
                    ),
                    "referrer": random.choice(
                        ["direct", "email", "social", "search_engine", "affiliate"]
                    ),
                }
            )

            session_counter += 1

            # Create a transaction from a converted session.
            if converted and transaction_counter < NUM_TRANSACTIONS:
                transaction_items = []
                valid_transaction = True

                for product_id, details in cart_contents.items():
                    quantity = details["quantity"]

                    if quantity <= 0:
                        continue

                    if inventory.update_stock(product_id, quantity):
                        transaction_items.append(
                            {
                                "product_id": product_id,
                                "quantity": quantity,
                                "unit_price": details["price"],
                                "subtotal": round(
                                    quantity * details["price"],
                                    2,
                                ),
                            }
                        )
                    else:
                        valid_transaction = False
                        break

                if valid_transaction and transaction_items:
                    subtotal = sum(
                        item["subtotal"] for item in transaction_items
                    )
                    discount = 0

                    if random.random() < 0.2:
                        discount_rate = random.choice(
                            [0.05, 0.1, 0.15, 0.2]
                        )
                        discount = round(subtotal * discount_rate, 2)

                    transactions.append(
                        {
                            "transaction_id": generate_transaction_id(),
                            "session_id": session_id,
                            "user_id": user["user_id"],
                            "timestamp": (
                                session_start
                                + datetime.timedelta(seconds=session_duration)
                            ).isoformat(),
                            "items": transaction_items,
                            "subtotal": subtotal,
                            "discount": discount,
                            "total": round(subtotal - discount, 2),
                            "payment_method": random.choice(
                                ["credit_card", "paypal", "apple_pay", "crypto"]
                            ),
                            "status": "completed",
                        }
                    )
                    transaction_counter += 1

        # Generate an additional transaction when needed.
        if (
            transaction_counter < NUM_TRANSACTIONS
            and random.random() < 0.2
        ):
            user = random.choice(users)
            selected_products = random.sample(
                products,
                k=min(3, len(products)),
            )

            transaction_items = []

            for product in selected_products:
                if product["is_active"]:
                    quantity = random.randint(1, 3)

                    if inventory.update_stock(product["product_id"], quantity):
                        transaction_items.append(
                            {
                                "product_id": product["product_id"],
                                "quantity": quantity,
                                "unit_price": product["base_price"],
                                "subtotal": round(
                                    quantity * product["base_price"],
                                    2,
                                ),
                            }
                        )

            if transaction_items:
                subtotal = sum(
                    item["subtotal"] for item in transaction_items
                )
                discount = 0

                if random.random() < 0.2:
                    discount_rate = random.choice([0.05, 0.1, 0.15, 0.2])
                    discount = round(subtotal * discount_rate, 2)

                transactions.append(
                    {
                        "transaction_id": generate_transaction_id(),
                        "session_id": None,
                        "user_id": user["user_id"],
                        "timestamp": fake.date_time_between(
                            start_date=f"-{TIMESPAN_DAYS}d",
                            end_date="now",
                        ).isoformat(),
                        "items": transaction_items,
                        "subtotal": subtotal,
                        "discount": discount,
                        "total": round(subtotal - discount, 2),
                        "payment_method": random.choice(
                            [
                                "credit_card",
                                "paypal",
                                "bank_transfer",
                                "gift_card",
                            ]
                        ),
                        "status": random.choice(
                            ["completed", "processing", "shipped", "delivered"]
                        ),
                    }
                )
                transaction_counter += 1

        if iteration % 10_000 == 0:
            print(
                f"Progress: {session_counter:,}/{NUM_SESSIONS:,} sessions, "
                f"{transaction_counter:,}/{NUM_TRANSACTIONS:,} transactions "
                f"(iteration {iteration:,})"
            )

    return sessions, transactions


# ---------------------------------------------------------------------------
# Dataset export
# ---------------------------------------------------------------------------

def save_json(filename, data):
    """Write data to a JSON file."""
    with open(filename, "w", encoding="utf-8") as output_file:
        json.dump(
            data,
            output_file,
            default=json_serializer,
            ensure_ascii=False,
        )


def save_datasets(users, products, categories, transactions, sessions, inventory):
    """Save all generated datasets to JSON files."""
    print("Saving datasets...")

    save_json("users.json", users)
    save_json("products.json", list(inventory.products.values()))
    save_json("categories.json", categories)
    save_json("transactions.json", transactions)

    for start in range(0, len(sessions), CHUNK_SIZE):
        chunk = sessions[start : start + CHUNK_SIZE]
        chunk_number = start // CHUNK_SIZE
        save_json(f"sessions_{chunk_number}.json", chunk)

    print(
        "\nDataset generation complete!\n"
        f"- Sessions: {len(sessions):,} (target: {NUM_SESSIONS:,})\n"
        f"- Transactions: {len(transactions):,} "
        f"(target: {NUM_TRANSACTIONS:,})\n"
        f"- Remaining products: "
        f"{sum(product['current_stock'] for product in inventory.products.values()):,}"
    )


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------

def main():
    print("Initializing dataset generation...")

    categories = generate_categories()
    products = generate_products(categories)
    users = generate_users()

    inventory = InventoryManager(products)

    sessions, transactions = generate_sessions_and_transactions(
        users,
        products,
        categories,
        inventory,
    )

    save_datasets(
        users,
        products,
        categories,
        transactions,
        sessions,
        inventory,
    )


if __name__ == "__main__":
    main()
