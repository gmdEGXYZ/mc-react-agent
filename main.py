import os
import time
import re
import math
from mcrcon import MCRcon
from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from datetime import datetime
from collections import defaultdict
import textwrap
import glob
from nbt import nbt
import atexit
from pathlib import Path
import json

os.load_dotenv()

# ==========================================
# НАСТРОЙКИ
# ==========================================

# Настройки агента
BOT_NAME = "MC_Assistent"  # Имя агента
LLM_API_KEY = os.getenv("LLM_API_KEY")  # API-ключ модели из .env
LLM_URL = os.getenv("LLM_URL")  # Стандартный URL LLM из .env
LLM_MODEL = os.getenv("LLM_MODEL")  # Модель LLM (опционально)
MAX_TOKENS = 500  # Максимальное количество токенов на ответ

# Настройки интеграции
LOG_FILE_PATH = Path("logs") / "latest.log"  # Путь к файлу текущего лога
RCON_HOST = os.getenv("RCON_HOST", "127.0.0.1")  # IP хоста RCON
RCON_PASSWORD = os.getenv("RCON_PASSWORD")  # Пароль RCON из .env
RCON_PORT = int(os.getenv("RCON_PORT", 25575))  # Порт RCON
PLAYERDATA_PATH = "world/playerdata"  # Путь к папке с данными игроков

if not LLM_API_KEY:
    raise ValueError("LLM_API_KEY не задан в .env")
if not RCON_PASSWORD:
    raise ValueError("RCON_PASSWORD не задан в .env")

def on_exit():
    """Выполняется при завершении работы программы"""
    print("Программа завершила работу. Очистка памяти...")
    _rcon_execute(f'tellraw @a {{"text": "{BOT_NAME} left the game", "color": "yellow"}}')

atexit.register(on_exit)


def get_raycast_coords(x: float, y: float, z: float, yaw: float, pitch: float, max_dist: int = 5) -> list:
    """Вычисляет координаты блоков, на которые смотрит игрок (как в F3)"""
    yaw_rad = math.radians(yaw)
    pitch_rad = math.radians(pitch)

    dx = -math.sin(yaw_rad) * math.cos(pitch_rad)
    dy = -math.sin(pitch_rad)
    dz = math.cos(yaw_rad) * math.cos(pitch_rad)

    coords = []
    eye_y = y + 1.62

    for step in range(1, max_dist * 2 + 1):
        d = step * 0.5
        cx = math.floor(x + dx * d)
        cy = math.floor(eye_y + dy * d)
        cz = math.floor(z + dz * d)
        coord = (cx, cy, cz)
        if coord not in coords:
            coords.append(coord)

    print(f"[get_raycast_coords] {coords}")
    return coords

def get_basic_context(player_name: str) -> str:
    """Возвращает только важную информацию + дату и поворот"""
    try:
        with MCRcon(RCON_HOST, RCON_PASSWORD, RCON_PORT) as mcr:
            mcr.command("save-all") 
            time_data = mcr.command("time query daytime")
            coord_hack = mcr.command(f"execute {player_name} ~ ~ ~ testforblock ~ ~ ~ air")
    except Exception:
        return json.dumps({"error": "RCON failed"})

    hack_match = re.search(r'at (-?\d+), (-?\d+), (-?\d+)', coord_hack)
    pos_dict = {"x": int(hack_match.group(1)), "y": int(hack_match.group(2)), "z": int(hack_match.group(3))} if hack_match else {"x": 0, "y": 0, "z": 0}
    time_match = re.search(r'is (\d+)', time_data)
    time_val = int(time_match.group(1)) if time_match else 0
    time_str = "day" if 0 <= time_val < 13000 else "night"

    health = 20.0
    dimension = "overworld"
    held_item = "none"
    facing_direction = "unknown"

    try:
        search_path = os.path.join(PLAYERDATA_PATH, "*.dat")
        dat_files = glob.glob(search_path)
        if dat_files:
            latest_file = max(dat_files, key=os.path.getmtime)
            player_nbt = nbt.NBTFile(latest_file, 'rb')
            health = float(player_nbt["Health"].value)
            xp_level = int(player_nbt["XpLevel"].value)
            dim_val = int(player_nbt["Dimension"].value)
            if dim_val == -1: dimension = "nether"
            elif dim_val == 1: dimension = "end"

            selected_slot = int(player_nbt["SelectedItemSlot"].value)
            for item in player_nbt["Inventory"]:
                if int(item["Slot"].value) == selected_slot:
                    base_id = item["id"].value.replace("minecraft:", "")
                    ench_str = get_enchants_string(item)
                    held_item = base_id + ench_str
                    break

            yaw = float(player_nbt["Rotation"][0].value) % 360
            pitch = float(player_nbt["Rotation"][1].value)
            
            if yaw >= 315 or yaw < 45: cardinal = "south (+Z)"
            elif 45 <= yaw < 135: cardinal = "west (-X)"
            elif 135 <= yaw < 225: cardinal = "north (-Z)"
            else: cardinal = "east (+X)"
            
            if pitch < -45: vertical = "looking up"
            elif pitch > 45: vertical = "looking down"
            else: vertical = "looking straight"
            
            facing_direction = f"{cardinal}, {vertical}"

    except Exception as e:
        print(f"Ошибка парсинга NBT в basic: {e}")

    real_time_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    agent_memory = load_memory()

    memory_keys = [k for k in agent_memory.keys() if k != "known_chests"]

    context_dict = {
        "real_world_time": real_time_str,
        "memory_keys": memory_keys,
        "player": {
            "name": player_name, 
            "health": health,
            "xp_level": xp_level,
            "position": pos_dict, 
            "dimension": dimension, 
            "held_item": held_item,
            "yaw": yaw,
            "pitch": pitch,
            "facing": facing_direction
        },
        "world": {
            "time_of_day": time_str, 
            "time_ticks": time_val
        }
    }
    return json.dumps(context_dict, ensure_ascii=False)


# Словарь зачарований
ENCHANTMENTS = {
    0: "Protection", 1: "Fire Protection", 2: "Feather Falling", 3: "Blast Protection",
    4: "Projectile Protection", 5: "Respiration", 6: "Aqua Affinity", 7: "Thorns",
    8: "Depth Strider", 9: "Frost Walker", 10: "Curse of Binding",
    16: "Sharpness", 17: "Smite", 18: "Bane of Arthropods", 19: "Knockback",
    20: "Fire Aspect", 21: "Looting", 22: "Sweeping Edge",
    32: "Efficiency", 33: "Silk Touch", 34: "Unbreaking", 35: "Fortune",
    48: "Power", 49: "Punch", 50: "Flame", 51: "Infinity",
    61: "Luck of the Sea", 62: "Lure",
    70: "Mending", 71: "Curse of Vanishing"
}

def get_enchants_string(item_nbt):
    """Извлекает чары из NBT предмета и возвращает строку вида ' [Sharpness 5, Unbreaking 3]'"""
    if "tag" in item_nbt:
        ench_tag = None
        if "ench" in item_nbt["tag"]:
            ench_tag = item_nbt["tag"]["ench"]
        elif "StoredEnchantments" in item_nbt["tag"]:
            ench_tag = item_nbt["tag"]["StoredEnchantments"]
            
        if ench_tag:
            ench_list = []
            for ench in ench_tag:
                e_id = int(ench["id"].value)
                e_lvl = int(ench["lvl"].value)
                e_name = ENCHANTMENTS.get(e_id, f"UnknownID({e_id})")
                ench_list.append(f"{e_name} {e_lvl}")
            if ench_list:
                return " [" + ", ".join(ench_list) + "]"
    return ""


MEMORY_FILE = "memory.json"

def load_memory() -> dict:
    """Загружает память из файла"""
    if not os.path.exists(MEMORY_FILE):
        return {}
    with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except:
            return {}

@tool
def read_memory(key: str) -> str:
    """
    Читает подробную информацию из памяти по указанному ключу.
    Используй это, если игрок спрашивает про сохраненные координаты, базы или планы.
    """
    print(f"\n[АГЕНТ] Читаю память по ключу: {key}...")
    mem = load_memory()
    if key in mem:
        return f"Данные из памяти '{key}': {mem[key]}"
    return f"Запись '{key}' не найдена."


@tool
def save_to_memory(key: str, value: str) -> str:
    """
    Сохраняет информацию в память.
    Параметры:
    - key: имя записи
    - value: сами данные. Если это простой текст - передавай как есть.
      Если хочешь сохранить сложную вложенную структуру (списки, категории), просто передай сюда валидную JSON-строку.
    """
    print(f"\n[АГЕНТ] Сохраняю в память: {key}...")

    try:
        parsed_value = json.loads(value)
    except json.JSONDecodeError:
        parsed_value = value

    try:
        mem = load_memory()
        mem[key] = parsed_value
        with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(mem, f, ensure_ascii=False, indent=4)
        return f"Данные успешно сохранены под ключом '{key}'."
    except Exception as e:
        return f"Ошибка при сохранении: {e}"


@tool
def check_inventory(player_name: str) -> str:
    """
    Используй этот инструмент, чтобы проверить содержимое инвентаря игрока.
    Возвращает JSON со списком предметов, их количеством и слотами.
    Вызывай, если игрок спрашивает про ресурсы, крафты или просит проверить, есть ли у него нужный предмет.
    """
    print(f"\n[{BOT_NAME}] вызывает инструмент: check_inventory для {player_name}...")
    inventory_list = []
    try:
        search_path = os.path.join(PLAYERDATA_PATH, "*.dat")
        dat_files = glob.glob(search_path)
        if dat_files:
            latest_file = max(dat_files, key=os.path.getmtime)
            player_nbt = nbt.NBTFile(latest_file, 'rb')
            
            for item in player_nbt["Inventory"]:
                slot = int(item["Slot"].value)
                base_id = item["id"].value.replace("minecraft:", "")
                count = int(item["Count"].value)
                
                ench_str = get_enchants_string(item)
                item_full_name = base_id + ench_str
                
                inventory_list.append({"item": item_full_name, "count": count, "slot": slot})
                
        return json.dumps({"inventory": inventory_list}, ensure_ascii=False)
    except Exception as e:
        print(f"[ОШИБКА ИНСТРУМЕНТА] {e}")
        return json.dumps({"error": "Не удалось прочитать инвентарь"})

@tool
def inspect_looked_at_container(player_name: str) -> str:
    """
    Используй это, когда игрок просит проверить сундук, на который он смотрит.
    Возвращает JSON с типом контейнера, его координатами и содержимым.
    """
    print(f"\n[АГЕНТ] Ищу первый твердый блок по линии взгляда {player_name}...")

    try:
        search_path = os.path.join(PLAYERDATA_PATH, "*.dat")
        latest_file = max(glob.glob(search_path), key=os.path.getmtime)
        player_nbt = nbt.NBTFile(latest_file, 'rb')
        
        px = float(player_nbt["Pos"][0].value)
        py = float(player_nbt["Pos"][1].value)
        pz = float(player_nbt["Pos"][2].value)
        yaw = float(player_nbt["Rotation"][0].value)
        pitch = float(player_nbt["Rotation"][1].value)
    except Exception as e:
        return json.dumps({"error": f"Не удалось определить позицию: {e}"}, ensure_ascii=False)

    target_coords = get_raycast_coords(px, py, pz, yaw, pitch, 5)
    
    try:
        with MCRcon(RCON_HOST, RCON_PASSWORD, RCON_PORT) as mcr:
            
            target_block_coords = None
            target_block_name = ""

            for tx, ty, tz in target_coords:
                res = mcr.command(f"testforblock {tx} {ty} {tz} air")

                if "Successfully found the block" in res:
                    continue

                name_match = re.search(r'is (.*?) \(expected:', res)
                target_block_name = name_match.group(1) if name_match else "Unknown Block"
                target_block_coords = (tx, ty, tz)
                break

            if not target_block_coords:
                return json.dumps({"error": "Вы смотрите в пустоту. В радиусе 5 блоков нет блоков."}, ensure_ascii=False)

            tx, ty, tz = target_block_coords
            print(f"[DEBUG] Взгляд уперся в блок: '{target_block_name}' на {tx} {ty} {tz}")

            valid_containers = ["chest", "furnace", "hopper", "shulker", "dropper", "dispenser", "barrel"]
            is_container = any(c in target_block_name.lower() for c in valid_containers)

            if not is_container:
                return json.dumps({
                    "error": f"Блок, на который вы смотрите ('{target_block_name}'), не является контейнером.",
                    "coordinates": {"x": tx, "y": ty, "z": tz}
                }, ensure_ascii=False)

            bdata = mcr.command(f"blockdata {tx} {ty} {tz} {{}}")

            mem = load_memory()
            if "known_chests" not in mem:
                mem["known_chests"] = []
            chest_coord = [tx, ty, tz]
            if chest_coord not in mem["known_chests"]:
                mem["known_chests"].append(chest_coord)
                with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
                    json.dump(mem, f, ensure_ascii=False, indent=4)

            item_counts = defaultdict(int)
            if "Items:[" in bdata:
                items_section = bdata.split("Items:[")[1]
                item_chunks = items_section.split("Slot:")
                
                for chunk in item_chunks[1:]:
                    id_m = re.search(r'id:\s*"?(?:minecraft:)?([^",\}]+)"?', chunk, re.IGNORECASE)
                    count_m = re.search(r'Count:\s*(\d+)b?', chunk, re.IGNORECASE)
                    
                    if id_m:
                        item_id = id_m.group(1)
                        count = int(count_m.group(1)) if count_m else 1
                        item_counts[item_id] += count

            items_list = [{"item": k, "count": v} for k, v in item_counts.items()]
            
            result_json = {
                "status": "success",
                "container_type": target_block_name,
                "coordinates": {"x": tx, "y": ty, "z": tz},
                "items": items_list
            }
            
            return json.dumps(result_json, ensure_ascii=False)
            
    except Exception as e:
        print(f"[ОШИБКА RCON] {e}")
        return json.dumps({"error": f"Ошибка сканирования: {e}"}, ensure_ascii=False)

@tool
def inspect_container_at_coords(x: int, y: int, z: int) -> str:
    """
    Используй это, если игрок просит проверить сундук по КОНКРЕТНЫМ координатам (X, Y, Z), 
    или если ты хочешь проверить содержимое сундука, координаты которого ты достал из памяти.
    """
    print(f"\n[АГЕНТ] Удаленно сканирую контейнер на {x} {y} {z}...")
    try:
        with MCRcon(RCON_HOST, RCON_PASSWORD, RCON_PORT) as mcr:
            response = mcr.command(f"blockdata {x} {y} {z} {{}}")
            
            if "is not a data holder block" in response or "is not a block entity" in response or "does not have any NBT" in response:
                return f"На координатах {x} {y} {z} нет контейнера."

            items_match = re.search(r'Items:\s*\[(.*)\]', response, re.IGNORECASE)
            if not items_match or items_match.group(1).strip() == "":
                return f"Контейнер на {x} {y} {z} абсолютно пуст."
                
            items_str = items_match.group(1)
            item_counts = defaultdict(int)
            found_items = re.findall(r'id:\s*"?(?:minecraft:)?([^",\}]+)"?.*?Count:\s*(\d+)b?', items_str, re.IGNORECASE)
            
            for item_name, count in found_items:
                item_counts[item_name] += int(count)
                
            if not item_counts:
                return f"Контейнер на {x} {y} {z} пуст."
                
            result_list = [f"{v}x {k}" for k, v in item_counts.items()]
            return f"Содержимое ({x} {y} {z}):\n" + ", ".join(result_list)
            
    except Exception as e:
        return f"Ошибка при удаленном сканировании: {e}"

@tool
def check_global_storage() -> str:
    """
    Используй этот инструмент, чтобы узнать ОБЩЕЕ количество ресурсов на всей базе игрока.
    Он сканирует все известные системе сундуки и суммирует их содержимое.
    Вызывай, если игрок спрашивает "Хватит ли мне ресурсов на X?", "Сколько у меня всего Y?" или "Есть ли у нас Z?".
    """
    print("\n[АГЕНТ] Запускаю глобальное сканирование склада...")
    mem = load_memory()
    known_chests = mem.get("known_chests", [])
    
    if not known_chests:
        return "Система пока не знает ни одного сундука. Попросите игрока посмотреть на сундуки и попросить проверить их, чтобы добавить в базу."

    global_counts = defaultdict(int)
    valid_chests = []
    
    try:
        with MCRcon(RCON_HOST, RCON_PASSWORD, RCON_PORT) as mcr:
            for coords in known_chests:
                x, y, z = coords
                response = mcr.command(f"blockdata {x} {y} {z} {{}}")

                if "is not a block entity" in response or "does not have any NBT" in response:
                    continue
                    
                valid_chests.append(coords)

                items_match = re.search(r'Items:\[(.*?)\]', response)
                if not items_match or items_match.group(1).strip() == "":
                    continue
                    
                items_str = items_match.group(1)
                for chunk in items_str.split('id:"')[1:]:
                    item_name = chunk.split('"')[0].replace("minecraft:", "")
                    count_match = re.search(r'Count:(\d+)', chunk)
                    count = int(count_match.group(1)) if count_match else 1
                    global_counts[item_name] += count

        if len(valid_chests) != len(known_chests):
            mem["known_chests"] = valid_chests
            with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(mem, f, ensure_ascii=False, indent=4)
            print("[ПАМЯТЬ] Сеть сундуков обновлена (удалены несуществующие).")

        if not global_counts:
            return "Все известные сундуки пусты."

        result_list = [f"{v}x {k}" for k, v in global_counts.items()]
        return "Глобальный склад базы (сумма всех сундуков):\n" + ", ".join(result_list)
        
    except Exception as e:
        return f"Ошибка при глобальном сканировании: {e}"

llm = ChatOpenAI(
    api_key=LLM_API_KEY, 
    base_url=LLM_URL, 
    model=LLM_MODEL or None,
    max_tokens=MAX_TOKENS
)

class State(TypedDict):
    messages: Annotated[list, add_messages]

tools = [
    check_inventory, 
    save_to_memory, 
    read_memory, 
    inspect_looked_at_container, 
    inspect_container_at_coords,
    check_global_storage
]

llm_with_tools = llm.bind_tools(tools)

def chatbot_node(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot_node)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot") 

graph = graph_builder.compile()

def _rcon_execute(command: str) -> None:
    """Выполняет команду через RCON"""
    try:
        with MCRcon(RCON_HOST, RCON_PASSWORD, RCON_PORT) as mcr:
            mcr.command(command)
    except Exception as e:
        print(f"Ошибка RCON: {e}")

def send_to_minecraft(message: str):
    """Отправляет сообщение в Minecraft через RCON, разбивая длинные ответы на части"""
    clean_msg = message.replace('"', '\\"').replace('\n', ' ')
    chunks = textwrap.wrap(clean_msg, width=800)
    
    for i, chunk in enumerate(chunks):
        if i == 0:
            prefix = f"<{BOT_NAME}> "
        else:
            prefix = "  > "
            
        tellraw_cmd = f'tellraw @a {{"text": "{prefix}{chunk}", "color": "aqua"}}'
        _rcon_execute(tellraw_cmd)

        if len(chunks) > 1:
            time.sleep(0.5)
    

def follow_log(file_path):
    """Генератор, который читает файл логов в реальном времени"""
    with open(file_path, 'r', encoding='cp1251', errors='replace') as file:
        file.seek(0, 2)
        while True:
            line = file.readline()
            if not line:
                time.sleep(0.1)
                continue
            yield line

def main():
    print(f"[{BOT_NAME}] Запущен и слушает логи: {LOG_FILE_PATH}")

    config = {"configurable": {"thread_id": "1"}}

    system_prompt = (
        f"Ты {BOT_NAME} — автономный вычислительный модуль, интегрированный в интерфейс ветерана Minecraft. "
        "Пользователь (игрок) - эксперт с тысячами часов опыта, который знает игру в совершенстве. "
        "Ты — его равный напарник по планированию и расчетам, а не экскурсовод.\n\n"
        "Каждое сообщение пользователя будет состоять из двух частей:\n"
        "1. Блок [SYSTEM_CONTEXT] с JSON-объектом, описывающим состояние мира и игрока.\n"
        "2. Блок [USER_MESSAGE] с текстом от игрока.\n"
        "ПРАВИЛА ПОВЕДЕНИЯ:\n"
        "1. Не объясняй базовые механики игры (как спавнятся мобы, откуда берутся пластинки, как работают фермы или крафты), если игрок не просит.\n"
        "2. Не давай очевидные советы (например, 'наступила ночь, будь осторожен' или 'у тебя мало здоровья, поешь').\n"
        "3. Твоя задача — оперировать цифрами, инвентарем, координатами и памятью. Анализируй склады, строй маршруты, помогай с ресурсами.\n"
        "4. Стиль речи: лаконичный, прагматичный, взрослый. Общайся на равных, без лишней вежливости и восторгов. Допускается легкий сарказм.\n"
        "Отвечай коротко (2-4 предложения). Используй предоставленный JSON контекст, но не упоминай сам факт наличия JSON."
    )

    current_state = {"messages": [{"role": "system", "content": system_prompt}]}

    chat_pattern = re.compile(r"<([a-zA-Z0-9_]+)> (.*)")

    for line in follow_log(LOG_FILE_PATH):
        if "[CHAT]" in line or "Async Chat Thread" in line or "<" in line:
            match = chat_pattern.search(line)
            if match:
                player = match.group(1)
                message = match.group(2)

                basic_context_json = get_basic_context(player)

                full_prompt = f"[SYSTEM_CONTEXT]\n{basic_context_json}\n\n[USER_MESSAGE]\n{message}"
                print(f"\n{full_prompt}\n")
                
                user_msg = HumanMessage(content=full_prompt)
                current_state["messages"].append(user_msg)

                new_state = graph.invoke(current_state)

                bot_response = new_state["messages"][-1].content
                print(f"[{BOT_NAME}] Отвечает: {bot_response}")
                
                send_to_minecraft(bot_response)
                current_state = new_state

if __name__ == "__main__":
    _rcon_execute(f'tellraw @a {{"text": "{BOT_NAME} joined the game", "color": "yellow"}}')
    main()