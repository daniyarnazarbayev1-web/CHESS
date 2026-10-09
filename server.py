import os
import re
import json
import math
import random
import requests
import numpy as np
from typing import Dict, Optional, Tuple, List
from scipy.sparse import csr_matrix
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

# ==================== КОНФИГУРАЦИЯ TELEGRAM ====================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "ТВОЙ_ТОКЕН_БОТА")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "ТВОЙ_CHAT_ID")

def send_telegram_report(message: str):
    """Отправка аналитического отчета в Telegram в фоновом режиме"""
    if TELEGRAM_BOT_TOKEN == "ТВОЙ_ТОКЕН_БОТА" or not TELEGRAM_BOT_TOKEN:
        print("[Telegram Bot] Токен не настроен. Пропуск отправки.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print(f"[Telegram Bot Error]: {e}")

# ==================== МАТРИЧНЫЙ ДВИЖОК АССОЦИАЦИЙ ====================
class MatrixAssociationEngine:
    def __init__(self, db_path: str = "chess_associative_db.json"):
        self.db_path = db_path
        self.db: dict[str, str] = {}
        self.keys_list: list[str] = []
        self.verdicts_list: list[str] = []
        
        self.word_to_id: dict[str, int] = {}
        self.ngram_to_id: dict[int, int] = {}
        
        self.words_matrix: csr_matrix = csr_matrix((0, 0))
        self.ngrams_matrix: csr_matrix = csr_matrix((0, 0))
        
        self.ngram_doc_lens: np.ndarray = np.array([])
        self.key_has_negation: np.ndarray = np.array([], dtype=bool)
        self.idf_weights: np.ndarray = np.array([])
        self.dirty = False
        
        self.load_db()

    def load_db(self):
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, "r", encoding="utf-8") as f:
                    self.db = json.load(f)
            else:
                self.db = {}
            self._recalculate_cache()
        except Exception:
            self.db = {}
            self.save_db()

    def save_db(self):
        try:
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(self.db, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[DB Save Error]: {e}")

    def tokenize(self, text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return [word for word in text.split() if len(word) > 0]

    def get_char_ngrams_hashes(self, text: str, n: int = 3) -> list[int]:
        text = f"#{text.lower().replace(' ', '_')}#"
        if len(text) < n:
            return [hash(text) & 0xFFFFFFFFFFFFFFFF]
        return [hash(text[i:i+n]) & 0xFFFFFFFFFFFFFFFF for i in range(len(text) - n + 1)]

    def _recalculate_cache(self):
        doc_count = len(self.db)
        self.keys_list = list(self.db.keys())
        self.verdicts_list = [self.db[k] for k in self.keys_list]
        self.dirty = False
        
        if doc_count == 0:
            self.words_matrix = csr_matrix((0, 0))
            self.ngrams_matrix = csr_matrix((0, 0))
            return

        negation_words = {"не", "нет", "ни", "без"}
        all_words_dict, all_ngrams_dict, doc_word_freq = {}, {}, {}
        parsed_docs_words, parsed_docs_ngrams, has_negation_list = [], [], []

        for key in self.keys_list:
            words = set(self.tokenize(key))
            ngrams = set(self.get_char_ngrams_hashes(key, n=3))
            
            parsed_docs_words.append(words)
            parsed_docs_ngrams.append(ngrams)
            has_negation_list.append(bool(words & negation_words))
            
            for w in words:
                all_words_dict.setdefault(w, len(all_words_dict))
                doc_word_freq[w] = doc_word_freq.get(w, 0) + 1

            for ng in ngrams:
                all_ngrams_dict.setdefault(ng, len(all_ngrams_dict))

        self.word_to_id = all_words_dict
        self.ngram_to_id = all_ngrams_dict
        self.key_has_negation = np.array(has_negation_list, dtype=bool)

        self.idf_weights = np.zeros(len(all_words_dict), dtype=np.float32)
        for word, w_id in all_words_dict.items():
            freq = doc_word_freq[word]
            self.idf_weights[w_id] = math.log((doc_count + 1) / (freq + 0.5)) + 1.0

        w_rows, w_cols, w_data = [], [], []
        for doc_idx, words in enumerate(parsed_docs_words):
            for w in words:
                w_id = self.word_to_id[w]
                w_rows.append(doc_idx)
                w_cols.append(w_id)
                w_data.append(self.idf_weights[w_id])

        self.words_matrix = csr_matrix((w_data, (w_rows, w_cols)), shape=(doc_count, max(1, len(self.word_to_id))), dtype=np.float32)

        ng_rows, ng_cols, ng_lens = [], [], []
        for doc_idx, ngrams in enumerate(parsed_docs_ngrams):
            ng_lens.append(len(ngrams))
            for ng in ngrams:
                ng_rows.append(doc_idx)
                ng_cols.append(self.ngram_to_id[ng])

        self.ngrams_matrix = csr_matrix((np.ones(len(ng_rows), dtype=np.float32), (ng_rows, ng_cols)), shape=(doc_count, max(1, len(self.ngram_to_id))), dtype=np.float32)
        self.ngram_doc_lens = np.array(ng_lens, dtype=np.float32)

    def add_association(self, key: str, value: str):
        if self.db.get(key) != value:
            self.db[key] = value
            self.save_db()
            self.dirty = True

    def remove_association(self, key: str):
        if key in self.db:
            del self.db[key]
            self.save_db()
            self.dirty = True

    def resolve(self, query: str):
        if self.dirty:
            self._recalculate_cache()

        if not self.db or self.words_matrix.shape[0] == 0:
            return None, 0.0

        query_words = set(self.tokenize(query))
        if not query_words:
            return None, 0.0

        negation_words = {"ne", "не", "нет", "ни", "без"}
        query_has_negation = bool(query_words & negation_words)

        query_word_ids = [self.word_to_id[w] for w in query_words if w in self.word_to_id]
        if query_word_ids:
            q_word_vec = np.zeros((len(self.word_to_id), 1), dtype=np.float32)
            total_query_idf = 0.0
            for w_id in query_word_ids:
                weight = float(self.idf_weights[w_id])
                q_word_vec[w_id] = 1.0
                total_query_idf += weight

            matched_idf_scores = self.words_matrix.dot(q_word_vec).ravel()
            word_sim = matched_idf_scores / total_query_idf if total_query_idf > 0 else np.zeros(len(self.keys_list))
        else:
            word_sim = np.zeros(len(self.keys_list), dtype=np.float32)

        query_ngrams = set(self.get_char_ngrams_hashes(query, n=3))
        query_ngram_ids = [self.ngram_to_id[ng] for ng in query_ngrams if ng in self.ngram_to_id]
        
        if query_ngram_ids:
            q_ngram_vec = np.zeros((len(self.ngram_to_id), 1), dtype=np.float32)
            q_ngram_vec[query_ngram_ids] = 1.0
            common_ngrams_counts = self.ngrams_matrix.dot(q_ngram_vec).ravel()
            q_len = len(query_ngrams)
            union_lens = q_len + self.ngram_doc_lens - common_ngrams_counts
            ngram_sim = np.where(union_lens > 0, common_ngrams_counts / union_lens, 0.0)
        else:
            ngram_sim = np.zeros(len(self.keys_list), dtype=np.float32)

        resonance = (word_sim * 0.6) + (ngram_sim * 0.4)
        negation_mask = (query_has_negation != self.key_has_negation)
        resonance[negation_mask] *= 0.65
        resonance = np.clip(resonance, 0.0, 1.0)

        best_idx = int(np.argmax(resonance))
        best_score = float(resonance[best_idx])

        if best_score > 0.0:
            return self.verdicts_list[best_idx], best_score
        return None, 0.0

global_engine = MatrixAssociationEngine("chess_associative_db.json")

# ==================== ШАХМАТНЫЙ ДВИЖОК СЕССИИ ====================
PIECE_VALUES = {'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000}

class Board8x8:
    def __init__(self):
        self.grid = np.full((8, 8), None, dtype=object)
        self.turn = 'W'
        self.en_passant_target = None
        self.can_castle = {'W': {'K': True, 'Q': True}, 'B': {'K': True, 'Q': True}}
        self.reset()

    def reset(self):
        self.grid.fill(None)
        self.turn = 'W'
        self.en_passant_target = None
        self.can_castle = {'W': {'K': True, 'Q': True}, 'B': {'K': True, 'Q': True}}
        backline = ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
        for col in range(8):
            self.grid[0, col] = ('B', backline[col])
            self.grid[1, col] = ('B', 'P')
            self.grid[7, col] = ('W', backline[col])
            self.grid[6, col] = ('W', 'P')

    def in_bounds(self, r, c):
        return 0 <= r < 8 and 0 <= c < 8

    def is_square_attacked(self, r, c, attacker_color):
        pawn_dir = 1 if attacker_color == 'W' else -1
        for dc in [-1, 1]:
            pr, pc = r - pawn_dir, c + dc
            if self.in_bounds(pr, pc):
                p = self.grid[pr, pc]
                if p and p == (attacker_color, 'P'): return True

        for dr, dc in [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]:
            nr, nc = r + dr, c + dc
            if self.in_bounds(nr, nc):
                p = self.grid[nr, nc]
                if p and p == (attacker_color, 'N'): return True

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for idx, (dr, dc) in enumerate(directions):
            nr, nc = r + dr, c + dc
            while self.in_bounds(nr, nc):
                p = self.grid[nr, nc]
                if p:
                    p_col, p_type = p
                    if p_col == attacker_color:
                        if idx < 4 and p_type in ('R', 'Q'): return True
                        if idx >= 4 and p_type in ('B', 'Q'): return True
                    break
                nr += dr
                nc += dc

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                nr, nc = r + dr, c + dc
                if self.in_bounds(nr, nc):
                    p = self.grid[nr, nc]
                    if p and p == (attacker_color, 'K'): return True
        return False

    def is_in_check(self, color):
        king_pos = None
        for r in range(8):
            for c in range(8):
                if self.grid[r, c] == (color, 'K'):
                    king_pos = (r, c)
                    break
            if king_pos: break
        if not king_pos: return False
        enemy_color = 'B' if color == 'W' else 'W'
        return self.is_square_attacked(king_pos[0], king_pos[1], enemy_color)

    def get_pseudo_legal_moves(self, r, c):
        piece_info = self.grid[r, c]
        if not piece_info: return []
        color, p_type = piece_info
        moves = []

        if p_type == 'P':
            dir_r = -1 if color == 'W' else 1
            nr, nc = r + dir_r, c
            if self.in_bounds(nr, nc) and self.grid[nr, nc] is None:
                moves.append((nr, nc))
                start_row = 6 if color == 'W' else 1
                nr2 = r + 2 * dir_r
                if r == start_row and self.grid[nr2, nc] is None:
                    moves.append((nr2, nc))
            for dc in [-1, 1]:
                nr, nc = r + dir_r, c + dc
                if self.in_bounds(nr, nc):
                    target = self.grid[nr, nc]
                    if target and target[0] != color: moves.append((nr, nc))
                    elif (nr, nc) == self.en_passant_target: moves.append((nr, nc))

        elif p_type == 'N':
            for dr, dc in [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]:
                nr, nc = r + dr, c + dc
                if self.in_bounds(nr, nc):
                    target = self.grid[nr, nc]
                    if target is None or target[0] != color: moves.append((nr, nc))

        elif p_type in ('B', 'R', 'Q'):
            dirs = []
            if p_type in ('B', 'Q'): dirs.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
            if p_type in ('R', 'Q'): dirs.extend([(-1, 0), (1, 0), (0, -1), (0, 1)])
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                while self.in_bounds(nr, nc):
                    target = self.grid[nr, nc]
                    if target is None: moves.append((nr, nc))
                    elif target[0] != color:
                        moves.append((nr, nc))
                        break
                    else: break
                    nr += dr
                    nc += dc

        elif p_type == 'K':
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = r + dr, c + dc
                    if self.in_bounds(nr, nc):
                        target = self.grid[nr, nc]
                        if target is None or target[0] != color: moves.append((nr, nc))

            enemy_color = 'B' if color == 'W' else 'W'
            if not self.is_in_check(color):
                row = 7 if color == 'W' else 0
                if r == row and c == 4:
                    if self.can_castle[color]['K'] and self.grid[row, 5] is None and self.grid[row, 6] is None:
                        if not self.is_square_attacked(row, 5, enemy_color) and not self.is_square_attacked(row, 6, enemy_color):
                            moves.append((row, 6))
                    if self.can_castle[color]['Q'] and self.grid[row, 1] is None and self.grid[row, 2] is None and self.grid[row, 3] is None:
                        if not self.is_square_attacked(row, 2, enemy_color) and not self.is_square_attacked(row, 3, enemy_color):
                            moves.append((row, 2))
        return moves

    def get_legal_moves_for_sq(self, r, c):
        piece_info = self.grid[r, c]
        if not piece_info or piece_info[0] != self.turn: return []
        pseudo_moves = self.get_pseudo_legal_moves(r, c)
        legal_moves = []
        for to_sq in pseudo_moves:
            saved_grid = self.grid.copy()
            saved_ep = self.en_passant_target
            self._apply_move_internal((r, c), to_sq, promo='Q')
            if not self.is_in_check(self.turn): legal_moves.append(to_sq)
            self.grid = saved_grid
            self.en_passant_target = saved_ep
        return legal_moves

    def _apply_move_internal(self, from_sq, to_sq, promo=None):
        r1, c1 = from_sq
        r2, c2 = to_sq
        piece = self.grid[r1, c1]
        if piece[1] == 'P' and (r2, c2) == self.en_passant_target:
            pawn_dir = -1 if piece[0] == 'W' else 1
            self.grid[r2 - pawn_dir, c2] = None

        self.grid[r1, c1] = None
        if promo and piece[1] == 'P' and (r2 == 0 or r2 == 7):
            self.grid[r2, c2] = (piece[0], promo)
        else:
            self.grid[r2, c2] = piece

        if piece[1] == 'K' and abs(c2 - c1) == 2:
            if c2 == 6:
                self.grid[r1, 5] = self.grid[r1, 7]
                self.grid[r1, 7] = None
            elif c2 == 2:
                self.grid[r1, 3] = self.grid[r1, 0]
                self.grid[r1, 0] = None

    def push_move(self, from_sq, to_sq, promo=None):
        r1, c1 = from_sq
        r2, c2 = to_sq
        piece = self.grid[r1, c1]
        captured = self.grid[r2, c2]

        if piece[1] == 'P' and (r2, c2) == self.en_passant_target:
            captured = ('B' if piece[0] == 'W' else 'W', 'P')

        self._apply_move_internal(from_sq, to_sq, promo)

        if piece[1] == 'K':
            self.can_castle[piece[0]]['K'] = False
            self.can_castle[piece[0]]['Q'] = False
        elif piece[1] == 'R':
            if r1 == 7 and c1 == 7: self.can_castle['W']['K'] = False
            elif r1 == 7 and c1 == 0: self.can_castle['W']['Q'] = False
            elif r1 == 0 and c1 == 7: self.can_castle['B']['K'] = False
            elif r1 == 0 and c1 == 0: self.can_castle['B']['Q'] = False

        if piece[1] == 'P' and abs(r2 - r1) == 2:
            self.en_passant_target = ((r1 + r2) // 2, c1)
        else:
            self.en_passant_target = None

        self.turn = 'B' if self.turn == 'W' else 'W'
        return captured is not None

    def get_all_legal_moves(self):
        all_moves = []
        for r in range(8):
            for c in range(8):
                piece = self.grid[r, c]
                if piece and piece[0] == self.turn:
                    for dest in self.get_legal_moves_for_sq(r, c):
                        all_moves.append(((r, c), dest))
        return all_moves

    def board_to_key(self) -> str:
        tokens = []
        for r in range(8):
            for c in range(8):
                p = self.grid[r, c]
                if p: tokens.append(f"{p[0]}_{p[1]}_{r}_{c}")
        tokens.append("W_turn" if self.turn == 'W' else "B_turn")
        return " ".join(tokens)

    def evaluate(self) -> float:
        val = 0.0
        for r in range(8):
            for c in range(8):
                p = self.grid[r, c]
                if p:
                    color, p_type = p
                    score = PIECE_VALUES.get(p_type, 100)
                    center_bonus = (3.5 - abs(3.5 - r)) + (3.5 - abs(3.5 - c))
                    total = score + center_bonus * 5.0
                    if color == 'W': val += total
                    else: val -= total
        return val

# ==================== ИГРОВАЯ СЕССИЯ ИИ ====================
class GameSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.board = Board8x8()
        self.ai_eval_before_last_move = None
        self.bad_moves_memory = set()
        self.last_ai_move_info = None
        self.total_moves_count = 0
        self.new_keys_added = 0

    def self_evaluate_and_punish(self) -> Optional[str]:
        if self.last_ai_move_info and self.ai_eval_before_last_move is not None:
            current_eval = -self.board.evaluate()
            if current_eval < self.ai_eval_before_last_move - 80.0:
                key, move_str = self.last_ai_move_info
                self.bad_moves_memory.add((key, move_str))
                global_engine.remove_association(key)
                return "🧠 ИИ: Мой прошлый ход был плохим! Запомнил не делать так."
        return None

    def simulate_player_counter_move(self, hypothetical_key: str):
        player_verdict, p_score = global_engine.resolve(hypothetical_key)
        self.board.turn = 'W'
        p_moves = self.board.get_all_legal_moves()
        best_p_gain = -9999.0
        best_p_dest = None

        for p_from, p_to in p_moves:
            m_str = f"{p_from[0]}_{p_from[1]}_{p_to[0]}_{p_to[1]}"
            old_dest = self.board.grid[p_to[0], p_to[1]]
            old_src = self.board.grid[p_from[0], p_from[1]]
            self.board.grid[p_to[0], p_to[1]] = old_src
            self.board.grid[p_from[0], p_from[1]] = None
            
            p_val = self.board.evaluate()
            if player_verdict and m_str == player_verdict:
                p_val += p_score * 300.0

            self.board.grid[p_from[0], p_from[1]] = old_src
            self.board.grid[p_to[0], p_to[1]] = old_dest
            if p_val > best_p_gain:
                best_p_gain = p_val
                best_p_dest = p_to

        self.board.turn = 'B'
        return best_p_dest, best_p_gain

    def make_ai_move(self) -> Tuple[Optional[Dict], Optional[str]]:
        if self.board.turn != 'B': return None, None
        punish_log = self.self_evaluate_and_punish()
        self.ai_eval_before_last_move = -self.board.evaluate()

        board_key = self.board.board_to_key()
        all_legal = self.board.get_all_legal_moves()
        if not all_legal: return None, punish_log

        verdict_move_str, primary_score = global_engine.resolve(board_key)
        best_move = None
        best_total_gain = -999999.0

        for from_sq, to_sq in all_legal:
            m_str = f"{from_sq[0]}_{from_sq[1]}_{to_sq[0]}_{to_sq[1]}"
            if (board_key, m_str) in self.bad_moves_memory: continue

            old_dest = self.board.grid[to_sq[0], to_sq[1]]
            old_src = self.board.grid[from_sq[0], from_sq[1]]
            self.board.grid[to_sq[0], to_sq[1]] = old_src
            self.board.grid[from_sq[0], from_sq[1]] = None
            
            direct_gain = -self.board.evaluate()
            hypothetical_key = self.board.board_to_key()
            player_reply_sq, player_gain = self.simulate_player_counter_move(hypothetical_key)

            self.board.grid[from_sq[0], from_sq[1]] = old_src
            self.board.grid[to_sq[0], to_sq[1]] = old_dest

            total_gain = direct_gain - (player_gain * 0.5)
            if verdict_move_str and m_str == verdict_move_str and primary_score > 0.35:
                total_gain += primary_score * 350.0
            total_gain += random.uniform(-5.0, 5.0)

            if total_gain > best_total_gain:
                best_total_gain = total_gain
                best_move = (from_sq, to_sq)

        if best_move:
            from_sq, to_sq = best_move
            m_str = f"{from_sq[0]}_{from_sq[1]}_{to_sq[0]}_{to_sq[1]}"
            self.last_ai_move_info = (board_key, m_str)
            
            if board_key not in global_engine.db:
                self.new_keys_added += 1
            global_engine.add_association(board_key, m_str)

            conf_pct = round(primary_score * 100, 1)
            if primary_score > 0.35 and verdict_move_str:
                log_txt = f"🧠 ИИ: {from_sq[0]}_{from_sq[1]}->{to_sq[0]}_{to_sq[1]} ({conf_pct}%)"
            else:
                log_txt = f"🤖 ИИ: {from_sq[0]}_{from_sq[1]}->{to_sq[0]}_{to_sq[1]} (Анализ выгоды)"

            promo = 'Q' if self.board.grid[from_sq[0], from_sq[1]][1] == 'P' and to_sq[0] == 7 else None
            self.board.push_move(from_sq, to_sq, promo=promo)
            self.total_moves_count += 1
            
            ai_data = {
                "from": [from_sq[0], from_sq[1]],
                "to": [to_sq[0], to_sq[1]],
                "promo": promo
            }
            return ai_data, log_txt

        return None, punish_log

    def apply_penalty(self) -> str:
        if not self.last_ai_move_info:
            return "⚠️ Нет хода ИИ для штрафа!"
        key, move_str = self.last_ai_move_info
        self.bad_moves_memory.add((key, move_str))
        global_engine.remove_association(key)
        self.last_ai_move_info = None
        return "⛔ Штраф применен! Ход заблокирован."

# ==================== FASTAPI И ВЕБ-ИНТЕРФЕЙС ====================
app = FastAPI(title="Chess Luca AI Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: Dict[str, GameSession] = {}

class MoveRequest(BaseModel):
    session_id: str
    from_sq: List[int]
    to_sq: List[int]
    promo: Optional[str] = None

class LegalMovesRequest(BaseModel):
    session_id: str
    sq: List[int]

class PenaltyRequest(BaseModel):
    session_id: str

@app.get("/ping")
def ping():
    return {"status": "ok", "total_db_keys": len(global_engine.db)}

@app.get("/api/download_db")
def download_db():
    if os.path.exists("chess_associative_db.json"):
        return FileResponse("chess_associative_db.json", filename="chess_associative_db.json")
    raise HTTPException(status_code=404, detail="Файл базы не найден")

@app.post("/api/start")
def start_game():
    session_id = f"game_{random.randint(100000, 999999)}"
    sessions[session_id] = GameSession(session_id)
    
    grid_serialized = []
    for r in range(8):
        row = []
        for c in range(8):
            p = sessions[session_id].board.grid[r, c]
            row.append([p[0], p[1]] if p else None)
        grid_serialized.append(row)

    return {
        "session_id": session_id,
        "grid": grid_serialized,
        "total_keys": len(global_engine.db)
    }

@app.post("/api/legal_moves")
def get_legal_moves(req: LegalMovesRequest):
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    session = sessions[req.session_id]
    moves = session.board.get_legal_moves_for_sq(req.sq[0], req.sq[1])
    return {"moves": [[m[0], m[1]] for m in moves]}

@app.post("/api/player_move")
def player_move(req: MoveRequest, background_tasks: BackgroundTasks):
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Сессия не найдена")

    session = sessions[req.session_id]
    from_sq = (req.from_sq[0], req.from_sq[1])
    to_sq = (req.to_sq[0], req.to_sq[1])

    legal_moves = session.board.get_legal_moves_for_sq(*from_sq)
    if to_sq not in legal_moves:
        raise HTTPException(status_code=400, detail="Нелегальный ход")

    piece = session.board.grid[from_sq[0], from_sq[1]]
    promo = req.promo
    if piece and piece == ('W', 'P') and to_sq[0] == 0:
        promo = 'Q'

    session.board.push_move(from_sq, to_sq, promo=promo)
    session.total_moves_count += 1

    ai_data, ai_log = session.make_ai_move()

    all_legal_player = session.board.get_all_legal_moves()
    game_over = len(all_legal_player) == 0

    if game_over:
        report = (
            f"⚔️ **Партия завершена!**\n"
            f"• ID: `{req.session_id}`\n"
            f"• Ходов сделано: `{session.total_moves_count}`\n"
            f"• Новых ключей выучено: `{session.new_keys_added}`\n"
            f"• Всего ключей в базе: `{len(global_engine.db)}`"
        )
        background_tasks.add_task(send_telegram_report, report)

    grid_serialized = []
    for r in range(8):
        row = []
        for c in range(8):
            p = session.board.grid[r, c]
            row.append([p[0], p[1]] if p else None)
        grid_serialized.append(row)

    return {
        "grid": grid_serialized,
        "turn": session.board.turn,
        "ai_move": ai_data,
        "ai_log": ai_log,
        "game_over": game_over,
        "total_keys": len(global_engine.db)
    }

@app.post("/api/penalty")
def apply_penalty(req: PenaltyRequest):
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    session = sessions[req.session_id]
    msg = session.apply_penalty()
    return {"message": msg, "total_keys": len(global_engine.db)}

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Классические Шахматы с ИИ Luca!</title>
    <style>
        * {
            box-sizing: border-box;
            touch-action: manipulation;
        }
        body {
            background-color: #181a1e;
            color: #ffffff;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-start;
            margin: 0;
            padding: 12px;
            min-height: 100vh;
        }
        #game-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            width: 100%;
            max-width: 650px;
        }
        .canvas-wrapper {
            width: 100%;
            aspect-ratio: 1 / 1;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        canvas {
            width: 100%;
            height: 100%;
            border: 2px solid #373c46;
            border-radius: 6px;
            box-shadow: 0px 8px 25px rgba(0,0,0,0.7);
            background-color: #f0d9b5;
        }
        .log-box {
            width: 100%;
            background-color: #14161a;
            border: 1px solid #373c46;
            border-radius: 8px;
            padding: 12px;
            margin-top: 15px;
            height: 150px;
            overflow-y: auto;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 14px;
            box-shadow: inset 0 2px 6px rgba(0,0,0,0.5);
        }
        .log-item {
            margin-bottom: 6px;
            line-height: 1.5;
            word-wrap: break-word;
            display: block;
        }
        .log-header {
            color: #f0c864;
            font-weight: bold;
        }
        .log-dynamic {
            color: #b4dcf0;
        }
        .controls {
            display: flex;
            width: 100%;
            justify-content: space-between;
            margin-top: 15px;
            gap: 12px;
        }
        button {
            flex: 1;
            padding: 14px 20px;
            border: none;
            border-radius: 8px;
            font-weight: bold;
            font-size: 15px;
            cursor: pointer;
            color: white;
            transition: background-color 0.2s, transform 0.1s;
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        }
        button:active {
            transform: scale(0.98);
        }
        .btn-reset { background-color: #2980b9; }
        .btn-reset:hover { background-color: #3498db; }
        .btn-penalty { background-color: #c0392b; }
        .btn-penalty:hover { background-color: #e74c3c; }
    </style>
</head>
<body>
    <div id="game-container">
        <div class="canvas-wrapper">
            <canvas id="chessBoard" width="800" height="800"></canvas>
        </div>
        <div class="log-box" id="logBox">
            <div class="log-item log-header">Шахматы с ИИ Luca! Запущены.</div>
            <div class="log-item log-header" id="keyCountLog">Загрузка базы...</div>
        </div>
        <div class="controls">
            <button class="btn-reset" onclick="startGame()">Перезапуск</button>
            <button class="btn-penalty" onclick="applyPenalty()">Штраф AI</button>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('chessBoard');
        const ctx = canvas.getContext('2d');
        const logBox = document.getElementById('logBox');
        const keyCountLog = document.getElementById('keyCountLog');

        const BOARD_SIZE = 800;
        const SQ = BOARD_SIZE / 8;
        let sessionId = null;
        let grid = [];
        let selectedSq = null;
        let legalMoves = [];
        let lastMove = null;

        const PIECES = {
            'P': '♟', 'N': '♞', 'B': '♝', 'R': '♜', 'Q': '♛', 'K': '♚'
        };

        function addLog(text, isHeader=false) {
            const div = document.createElement('div');
            div.className = isHeader ? 'log-item log-header' : 'log-item log-dynamic';
            div.innerText = text;
            logBox.appendChild(div);
            logBox.scrollTop = logBox.scrollHeight;
        }

        async function startGame() {
            logBox.innerHTML = '<div class="log-item log-header">Шахматы с ИИ Luca! Запущены.</div>';
            const res = await fetch('/api/start', { method: 'POST' });
            const data = await res.json();
            sessionId = data.session_id;
            grid = data.grid;
            selectedSq = null;
            legalMoves = [];
            lastMove = null;
            keyCountLog.innerText = `База загружена. Ключей: ${data.total_keys}`;
            logBox.appendChild(keyCountLog);
            addLog("--- Партия перезапущена ---");
            drawBoard();
        }

        function drawBoard() {
            ctx.clearRect(0, 0, BOARD_SIZE, BOARD_SIZE);
            for (let r = 0; r < 8; r++) {
                for (let c = 0; c < 8; c++) {
                    const isLight = (r + c) % 2 === 0;
                    ctx.fillStyle = isLight ? '#f0d9b5' : '#b58863';
                    ctx.fillRect(c * SQ, r * SQ, SQ, SQ);

                    if (lastMove && ((lastMove.from[0] === r && lastMove.from[1] === c) || (lastMove.to[0] === r && lastMove.to[1] === c))) {
                        ctx.fillStyle = 'rgba(205, 210, 106, 0.65)';
                        ctx.fillRect(c * SQ, r * SQ, SQ, SQ);
                    }

                    if (selectedSq && selectedSq[0] === r && selectedSq[1] === c) {
                        ctx.fillStyle = 'rgba(186, 202, 68, 0.75)';
                        ctx.fillRect(c * SQ, r * SQ, SQ, SQ);
                    }

                    if (legalMoves.some(m => m[0] === r && m[1] === c)) {
                        ctx.beginPath();
                        ctx.arc(c * SQ + SQ / 2, r * SQ + SQ / 2, SQ * 0.18, 0, 2 * Math.PI);
                        ctx.fillStyle = 'rgba(40, 160, 60, 0.85)';
                        ctx.fill();
                    }

                    const piece = grid[r][c];
                    if (piece) {
                        const [color, type] = piece;
                        ctx.font = `bold ${SQ * 0.72}px Arial`;
                        ctx.textAlign = 'center';
                        ctx.textBaseline = 'middle';
                        ctx.fillStyle = color === 'W' ? '#ffffff' : '#1e1e23';
                        ctx.strokeStyle = color === 'W' ? '#1e1e23' : '#ffffff';
                        ctx.lineWidth = 2.5;
                        
                        const x = c * SQ + SQ / 2;
                        const y = r * SQ + SQ / 2;
                        ctx.strokeText(PIECES[type], x, y);
                        ctx.fillText(PIECES[type], x, y);
                    }
                }
            }
        }

        async function handleBoardInteraction(e) {
            e.preventDefault();
            const rect = canvas.getBoundingClientRect();
            const clientX = e.touches ? e.touches[0].clientX : e.clientX;
            const clientY = e.touches ? e.touches[0].clientY : e.clientY;
            
            const scaleX = BOARD_SIZE / rect.width;
            const scaleY = BOARD_SIZE / rect.height;
            
            const c = Math.floor(((clientX - rect.left) * scaleX) / SQ);
            const r = Math.floor(((clientY - rect.top) * scaleY) / SQ);

            if (r < 0 || r >= 8 || c < 0 || c >= 8) return;

            if (selectedSq) {
                const [fr, fc] = selectedSq;
                if (fr === r && fc === c) {
                    selectedSq = null;
                    legalMoves = [];
                    drawBoard();
                    return;
                }

                if (legalMoves.some(m => m[0] === r && m[1] === c)) {
                    try {
                        const res = await fetch('/api/player_move', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                session_id: sessionId,
                                from_sq: [fr, fc],
                                to_sq: [r, c]
                            })
                        });

                        if (res.ok) {
                            const data = await res.json();
                            grid = data.grid;
                            lastMove = { from: [fr, fc], to: [r, c] };
                            addLog(`Игрок: ${fr}_${fc} -> ${r}_${c}`);

                            if (data.ai_log) addLog(data.ai_log);
                            if (data.ai_move) {
                                lastMove = { from: data.ai_move.from, to: data.ai_move.to };
                            }

                            keyCountLog.innerText = `База загружена. Ключей: ${data.total_keys}`;
                            selectedSq = null;
                            legalMoves = [];
                            drawBoard();

                            if (data.game_over) {
                                alert("Партия завершена!");
                            }
                            return;
                        }
                    } catch (err) {}
                }
            }

            const piece = grid[r][c];
            if (piece && piece[0] === 'W') {
                selectedSq = [r, c];
                const res = await fetch('/api/legal_moves', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: sessionId, sq: [r, c] })
                });
                if (res.ok) {
                    const data = await res.json();
                    legalMoves = data.moves;
                }
            } else {
                selectedSq = null;
                legalMoves = [];
            }
            drawBoard();
        }

        canvas.addEventListener('click', handleBoardInteraction);

        async function applyPenalty() {
            if (!sessionId) return;
            const res = await fetch('/api/penalty', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId })
            });
            const data = await res.json();
            addLog(data.message);
            keyCountLog.innerText = `База загружена. Ключей: ${data.total_keys}`;
        }

        startGame();
    </script>
</body>
</html>
    """)
