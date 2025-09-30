#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Транспортная задача: 3 фабрики × 5 магазинов (но код работает для любых m×n).

Методы:
- Северо-западный угол
- Минимального элемента
- Фогеля
- Потенциалов (MODI) с пошаговым выводом

Выход оформлен в виде «тетрадного» представления: над столбцами стоят b_j,
слева от строк стоят a_i; в каждой клетке сверху — тариф c_ij, снизу — отгрузка x_ij.
Также печатаются потенциалы u_i, v_j и матрица ΔC = C - (u ⊕ v).

Как использовать:
1) Отредактируйте блок INPUT_DATA ниже.
2) Запустите:  python3 transport.py
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set
import math
import itertools
import copy

# ======== INPUT_DATA ========
# Пример задания
SUPPLY = [200, 450, 250]
DEMAND = [100, 125, 325, 250, 100]
COSTS = [
    [5, 8, 7, 10, 3],
    [4, 2, 2, 5, 6],
    [7, 3, 5, 9, 2],
]
# ============================

Number = int

@dataclass
class Solution:
    x: List[List[int]]                 # план перевозок
    basis: Set[Tuple[int,int]]         # базисные клетки (включая нулевые, если они добавлены искусственно)

@dataclass
class Potentials:
    u: List[Optional[int]]
    v: List[Optional[int]]


def deep_zero_matrix(m: int, n: int) -> List[List[int]]:
    return [[0 for _ in range(n)] for _ in range(m)]


def total_cost(C: List[List[int]], X: List[List[int]]) -> int:
    return sum(C[i][j] * X[i][j] for i in range(len(C)) for j in range(len(C[0])))


def balance_problem(a: List[int], b: List[int], C: List[List[int]]):
    a_sum, b_sum = sum(a), sum(b)
    A = a[:]
    B = b[:]
    C2 = [row[:] for row in C]
    if a_sum == b_sum:
        return A, B, C2, None
    if a_sum > b_sum:
        # добавить фиктивный столбец (потребитель)
        diff = a_sum - b_sum
        for row in C2:
            row.append(0)
        B = B + [diff]
        return A, B, C2, ("demand", diff)
    else:
        diff = b_sum - a_sum
        C2.append([0]*len(B))
        A = A + [diff]
        return A, B, C2, ("supply", diff)


# ---------- Рисование таблиц «как на листе» ----------

def _format_F_formula(C: List[List[int]], X: List[List[int]]) -> str:
    terms: List[str] = []
    for i in range(len(C)):
        for j in range(len(C[0])):
            x = X[i][j]
            if x:
                terms.append(f"{x}*{C[i][j]}")
    return "F = " + " + ".join(terms) + f" = {total_cost(C, X)}"


def draw_table(C: List[List[int]], X: List[List[int]], a: List[int], b: List[int],
               title: str,
               potentials: Optional[Potentials] = None,
               deltas: Optional[List[List[Optional[int]]]] = None,
               show_zero_alloc: bool = False,
               pretty: bool = True,
               show_formula: bool = True) -> str:
    m, n = len(a), len(b)
    LEFT = 10
    W = 9  # ширина ячейки
    lines: List[str] = []
    lines.append(title)

    def hline(left_char:str='+-', mid_char:str='+-', right_char:str='-+'):
        return "+" + "-"*(LEFT) + "+" + "+".join(["-"*W for _ in range(n)]) + "+"

    # шапка b_j и b значения
    header1 = f"{'Базис':<{LEFT}}|" + "|".join(f"{('b'+str(j+1)):^{W}}" for j in range(n)) + "|"
    header2 = f"{'':<{LEFT}}|" + "|".join(f"{b[j]:^{W}}" for j in range(n)) + "|"
    lines.append(hline())
    lines.append(header1)
    lines.append(header2)
    lines.append(hline())

    for i in range(m):
        left = f"a{i+1}={a[i]:>4}"
        row_c = f"{left:<{LEFT}}|" + "|".join(f"{C[i][j]:^{W}}" for j in range(n)) + "|"
        if potentials is not None and potentials.u:
            uval = potentials.u[i]
            row_c += f"   u{i+1}={uval if uval is not None else '-'}"
        lines.append(row_c)
        row_x = f"{'':<{LEFT}}|"
        for j in range(n):
            xij = X[i][j]
            cell = f"({xij})" if (xij != 0 or show_zero_alloc) else ""
            row_x += f"{cell:^{W}}|"
        lines.append(row_x)
        lines.append(hline())
    if potentials is not None and potentials.v:
        vline = f"{'v:':<{LEFT}}|" + "|".join(
            f"{(potentials.v[j] if potentials.v[j] is not None else '-'):^{W}}" for j in range(n)
        ) + "|"
        lines.append(vline)
    F = total_cost(C, X)
    lines.append(f"F = {F}")
    if show_formula:
        lines.append(_format_F_formula(C, X))

    if deltas is not None:
        lines.append("ΔC = C - (u⊕v):")
        # печать дельт в виде таблицы
        lines.append(hline())
        drow1 = f"{'':<{LEFT}}|" + "|".join(f"{('Cij-u_i-v_j'):^{W}}" for _ in range(n)) + "|"  # заглушка шапки
        # не нужна шапка формулы в каждой колонке — сразу значения
        for i in range(m):
            dline = f"{'':<{LEFT}}|" + "|".join(
                f"{(deltas[i][j] if deltas[i][j] is not None else '-'):^{W}}" for j in range(n)
            ) + "|"
            lines.append(dline)
            lines.append(hline())
    return "\n".join(lines)


# ---------- Вспомогательные для базиса и потенциалов ----------

def basis_from_allocation(X: List[List[int]]) -> Set[Tuple[int,int]]:
    m, n = len(X), len(X[0])
    B: Set[Tuple[int,int]] = set()
    for i in range(m):
        for j in range(n):
            if X[i][j] > 0:
                B.add((i,j))
    return B

class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0]*n
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1
        return True


def ensure_tree_basis(m: int, n: int, C: List[List[int]], basis: Set[Tuple[int,int]]) -> Set[Tuple[int,int]]:
    # строим остов (m+n узлов: R_i и C_j). добавляем нулевые клетки, соединяющие разные компоненты
    total_nodes = m + n
    while True:
        dsu = DSU(total_nodes)
        for (i,j) in basis:
            dsu.union(i, m + j)
        comps = {dsu.find(k) for k in range(total_nodes)}
        edges_needed = (m + n - 1) - len(basis)
        if len(comps) == 1 and edges_needed <= 0:
            # уже одно дерево с нужным числом рёбер
            break
        # ищем минимальную по стоимости клетку, соединяющую разные компоненты и не входящую в базис
        best = None
        best_cost = math.inf
        for i in range(m):
            for j in range(n):
                if (i,j) in basis:
                    continue
                if dsu.find(i) != dsu.find(m + j):
                    if C[i][j] < best_cost:
                        best_cost = C[i][j]
                        best = (i,j)
        if best is None:
            # если ничего не нашли (редко), добавим любую небазисную минимальную клетку
            for i in range(m):
                for j in range(n):
                    if (i,j) not in basis:
                        if C[i][j] < best_cost:
                            best_cost = C[i][j]
                            best = (i,j)
        if best is None:
            break
        basis.add(best)
    return basis


def compute_potentials(m: int, n: int, C: List[List[int]], basis: Set[Tuple[int,int]]) -> Potentials:
    # граф между R_i и C_j по базисным клеткам
    adj: Dict[str, List[Tuple[str,int,int]]] = {}
    def add_edge(ri: int, cj: int, cij: int):
        rnode = f"R{ri}"
        cnode = f"C{cj}"
        adj.setdefault(rnode, []).append((cnode, ri, cj))
        adj.setdefault(cnode, []).append((rnode, ri, cj))
    for (i,j) in basis:
        add_edge(i,j,C[i][j])
    u = [None]*m
    v = [None]*n
    # обходим компоненты, фиксируя одну вершину в 0
    visited: Set[str] = set()
    for start in list(adj.keys()):
        if start in visited:
            continue
        # если старт узел — строка или столбец
        if start[0] == 'R':
            si = int(start[1:])
            u[si] = 0
        else:
            sj = int(start[1:])
            v[sj] = 0
        stack = [start]
        visited.add(start)
        while stack:
            node = stack.pop()
            for nxt, i, j in adj.get(node, []):
                if nxt in visited:
                    continue
                if node[0] == 'R':
                    # u_i + v_j = c_ij
                    # node=R, nxt=C
                    ri = int(node[1:])
                    u_i = u[ri]
                    v[j] = C[ri][j] - (u_i if u_i is not None else 0)
                else:
                    # node=C, nxt=R
                    cj = int(node[1:])
                    v_j = v[cj]
                    u[i] = C[i][cj] - (v_j if v_j is not None else 0)
                visited.add(nxt)
                stack.append(nxt)
    return Potentials(u=u, v=v)


def reduced_costs(C: List[List[int]], pots: Potentials) -> List[List[Optional[int]]]:
    m, n = len(C), len(C[0])
    R: List[List[Optional[int]]] = [[None]*n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            if pots.u[i] is None or pots.v[j] is None:
                R[i][j] = None
            else:
                R[i][j] = C[i][j] - pots.u[i] - pots.v[j]
    return R


# ---------- Начальные методы ----------

def method_northwest(a: List[int], b: List[int], C: List[List[int]]) -> Solution:
    m, n = len(a), len(b)
    s = a[:]
    d = b[:]
    X = deep_zero_matrix(m, n)
    i = j = 0
    while i < m and j < n:
        if s[i] == 0:
            i += 1
            continue
        if d[j] == 0:
            j += 1
            continue
        t = min(s[i], d[j])
        X[i][j] = t
        s[i] -= t
        d[j] -= t
        if s[i] == 0 and d[j] == 0:
            # двигаемся вправо (типичный выбор), вторую клетку оставим на исправление базиса
            if j+1 < n:
                j += 1
            else:
                i += 1
        elif s[i] == 0:
            i += 1
        else:
            j += 1
    B = basis_from_allocation(X)
    B = ensure_tree_basis(m, n, C, B)
    return Solution(x=X, basis=B)


def method_min_cost(a: List[int], b: List[int], C: List[List[int]]) -> Solution:
    m, n = len(a), len(b)
    s = a[:]
    d = b[:]
    X = deep_zero_matrix(m, n)
    cells = sorted(((C[i][j], i, j) for i in range(m) for j in range(n)))
    for _, i, j in cells:
        if s[i] == 0 or d[j] == 0:
            continue
        t = min(s[i], d[j])
        X[i][j] = t
        s[i] -= t
        d[j] -= t
        if sum(s) == 0 or sum(d) == 0:
            break
    B = basis_from_allocation(X)
    B = ensure_tree_basis(m, n, C, B)
    return Solution(x=X, basis=B)


def method_vogel(a: List[int], b: List[int], C: List[List[int]]) -> Solution:
    m, n = len(a), len(b)
    s = a[:]
    d = b[:]
    X = deep_zero_matrix(m, n)
    active_rows = set(range(m))
    active_cols = set(range(n))

    def row_penalty(i: int) -> int:
        costs = [C[i][j] for j in active_cols if d[j] > 0]
        if len(costs) == 0:
            return -1
        if len(costs) == 1:
            return costs[0]
        sc = sorted(costs)
        return sc[1] - sc[0]

    def col_penalty(j: int) -> int:
        costs = [C[i][j] for i in active_rows if s[i] > 0]
        if len(costs) == 0:
            return -1
        if len(costs) == 1:
            return costs[0]
        sc = sorted(costs)
        return sc[1] - sc[0]

    while active_rows and active_cols and (sum(s) > 0 and sum(d) > 0):
        # вычислить штрафы
        row_p = {i: row_penalty(i) for i in list(active_rows)}
        col_p = {j: col_penalty(j) for j in list(active_cols)}
        # выбрать максимальный штраф; при равенстве — где минимальный тариф
        best_kind = None
        best_idx = None
        best_pen = -1
        best_min_cost = math.inf
        # строки
        for i in list(active_rows):
            pen = row_p[i]
            if pen < 0:
                continue
            minc = min((C[i][j] for j in active_cols if d[j] > 0), default=math.inf)
            key = (pen, -minc)  # max pen, then min cost
            if pen > best_pen or (pen == best_pen and minc < best_min_cost):
                best_pen = pen
                best_kind = 'row'
                best_idx = i
                best_min_cost = minc
        # столбцы
        for j in list(active_cols):
            pen = col_p[j]
            if pen < 0:
                continue
            minc = min((C[i][j] for i in active_rows if s[i] > 0), default=math.inf)
            if pen > best_pen or (pen == best_pen and minc < best_min_cost):
                best_pen = pen
                best_kind = 'col'
                best_idx = j
                best_min_cost = minc
        if best_kind is None:
            break
        if best_kind == 'row':
            i = best_idx
            # выбрать min тариф в строке
            candidates = [(C[i][j], j) for j in active_cols if d[j] > 0]
            candidates.sort()
            j = candidates[0][1]
        else:
            j = best_idx
            candidates = [(C[i][j], i) for i in active_rows if s[i] > 0]
            candidates.sort()
            i = candidates[0][1]
        t = min(s[i], d[j])
        X[i][j] += t
        s[i] -= t
        d[j] -= t
        if s[i] == 0:
            active_rows.discard(i)
        if d[j] == 0:
            active_cols.discard(j)
    B = basis_from_allocation(X)
    B = ensure_tree_basis(m, n, C, B)
    return Solution(x=X, basis=B)


# ---------- MODI (метод потенциалов) ----------

def find_cycle_in_basis(m: int, n: int, basis: Set[Tuple[int,int]], enter: Tuple[int,int]) -> List[Tuple[int,int]]:
    # базис — дерево; добавим enter и найдём уникальный цикл
    # представим вершины как R_i (0..m-1) и C_j (0..n-1)
    # найдём путь между R_i и C_j в дереве
    i0, j0 = enter
    # строим граф на основе basis
    graph: Dict[str, List[str]] = {}
    def add_edge(ri: int, cj: int):
        rnode = f"R{ri}"
        cnode = f"C{cj}"
        graph.setdefault(rnode, []).append(cnode)
        graph.setdefault(cnode, []).append(rnode)
    for (i,j) in basis:
        add_edge(i,j)
    start = f"R{i0}"
    target = f"C{j0}"
    # BFS
    from collections import deque
    q = deque([start])
    prev: Dict[str, Optional[str]] = {start: None}
    while q:
        u = q.popleft()
        if u == target:
            break
        for v in graph.get(u, []):
            if v not in prev:
                prev[v] = u
                q.append(v)
    if target not in prev:
        # если по какой-то причине нет пути (не должно быть), вернуть пусто
        return []
    # восстановим путь узлов
    nodes_path: List[str] = []
    cur = target
    while cur is not None:
        nodes_path.append(cur)
        cur = prev[cur]
    nodes_path.reverse()  # R_i0 ... C_j0
    # преобразуем в рёбра (клетки)
    edges: List[Tuple[int,int]] = []
    for k in range(len(nodes_path)-1):
        u, v = nodes_path[k], nodes_path[k+1]
        if u[0] == 'R':
            i = int(u[1:]); j = int(v[1:])  # u=R, v=C
        else:
            i = int(v[1:]); j = int(u[1:])  # u=C, v=R
        edges.append((i,j))
    # добавим входящую дугу в начало, чтобы чередовать знаки + - + -
    cycle = [enter] + edges
    return cycle


def modi_optimize(a: List[int], b: List[int], C: List[List[int]], sol: Solution,
                  verbose_steps: bool = True) -> Tuple[Solution, List[str]]:
    m, n = len(a), len(b)
    X = copy.deepcopy(sol.x)
    basis = set(sol.basis)
    log_blocks: List[str] = []
    step = 1
    while True:
        # обеспечить «дерево»
        basis = ensure_tree_basis(m, n, C, basis)
        pots = compute_potentials(m, n, C, basis)
        R = reduced_costs(C, pots)
        # найдём самый отрицательный ΔC вне базиса
        best_cell = None
        best_val = 0
        for i in range(m):
            for j in range(n):
                if (i,j) in basis:
                    continue
                rij = R[i][j]
                if rij is not None and rij < best_val:
                    best_val = rij
                    best_cell = (i,j)
        # печать текущего шага
        block_title = f"Метод потенциалов — шаг {step}"
        log_blocks.append(draw_table(C, X, a, b, block_title, pots, R, show_zero_alloc=False, pretty=True, show_formula=True))
        log_blocks.append("")
        if best_cell is None:
            # оптимально
            break
        # строим цикл
        cycle = find_cycle_in_basis(m, n, basis, best_cell)
        if not cycle:
            # на всякий случай — добавим и попробуем ещё раз
            basis.add(best_cell)
            cycle = find_cycle_in_basis(m, n, basis, best_cell)
            if not cycle:
                # не удалось построить цикл — выходим
                break
        # определяем знак рёбер цикла: +, -, +, -, ... (включая входящее ребро как +)
        signs = []
        for k, cell in enumerate(cycle):
            signs.append(1 if k % 2 == 0 else -1)
        # среди рёбер с отрицательным знаком берём минимальную отгрузку как θ
        theta = math.inf
        minus_cells = []
        for k, (i,j) in enumerate(cycle):
            if signs[k] == -1:
                minus_cells.append((i,j))
                theta = min(theta, X[i][j])
        if theta == math.inf:
            # теоретически не должно
            break
        # выполняем перестановку
        for k, (i,j) in enumerate(cycle):
            if signs[k] == 1:
                X[i][j] += theta
            else:
                X[i][j] -= theta
        # обновляем базис: добавить входящую; удалить одну из минус-клеток, ставшей нулевой
        basis.add(best_cell)
        # кандидаты на удаление
        candidates = [cell for cell in minus_cells if X[cell[0]][cell[1]] == 0]
        removed = False
        for cand in candidates:
            new_basis = set(basis)
            if cand in new_basis:
                new_basis.remove(cand)
            basis = new_basis
            removed = True
            break
        if not removed:
            # если никто не занулился (редкий случай), удалим любой минус-элемент, чтобы разрушить цикл
            if minus_cells:
                basis.discard(minus_cells[0])
        step += 1
    # финальный блок с планом
    final_title = "Метод потенциалов — итоговый план"
    pots = compute_potentials(m, n, C, basis)
    R = reduced_costs(C, pots)
    block = draw_table(C, X, a, b, final_title, pots, R, pretty=True, show_formula=True)
    # найдём альтернативные оптимальные клетки (ΔC=0 вне базиса)
    alt_zeros: List[Tuple[int,int]] = []
    for i in range(m):
        for j in range(n):
            if (i,j) not in basis and R[i][j] == 0:
                alt_zeros.append((i+1,j+1))
    if alt_zeros:
        block += "\nАльтернативные оптимальные клетки (вне базиса, ΔC=0): " + \
                 ", ".join([f"(i={i}, j={j})" for i,j in alt_zeros])
    log_blocks.append(block)
    return Solution(x=X, basis=basis), log_blocks


# ---------- Высокоуровневая процедура ----------

def solve_and_print(a0: List[int], b0: List[int], C0: List[List[int]]):
    # балансировка
    a, b, C, bal = balance_problem(a0, b0, C0)
    if bal is not None:
        kind, diff = bal
        print("Внимание: задача несбалансирована. Добавлена фиктивная", "строка" if kind=='supply' else "колонка", f"на {diff} единиц с нулевыми тарифами.")
        print()

    m, n = len(a), len(b)

    # начальные решения
    sol_nw = method_northwest(a, b, C)
    sol_mc = method_min_cost(a, b, C)
    sol_vg = method_vogel(a, b, C)

    print(draw_table(C, sol_nw.x, a, b, "Метод северо-западного угла", pretty=True, show_formula=True))
    print()
    print(draw_table(C, sol_mc.x, a, b, "Метод минимального элемента", pretty=True, show_formula=True))
    print()
    print(draw_table(C, sol_vg.x, a, b, "Метод Фогеля", pretty=True, show_formula=True))
    print()

    # выбрать лучшее из трёх
    costs = [
        (total_cost(C, sol_nw.x), 'NW', sol_nw),
        (total_cost(C, sol_mc.x), 'MIN', sol_mc),
        (total_cost(C, sol_vg.x), 'VOGEL', sol_vg),
    ]
    costs.sort()
    best_cost, best_name, best_sol = costs[0]
    print(f"Выбрано начальное решение для метода потенциалов: {best_name} (стоимость {best_cost})")
    print()

    # метод потенциалов
    final_sol, logs = modi_optimize(a, b, C, best_sol, verbose_steps=True)
    for block in logs:
        print(block)
        print()

    # вывод вектором x_опт (по строкам)
    x_vec: List[int] = []
    for i in range(m):
        for j in range(n):
            x_vec.append(final_sol.x[i][j])
    print("Оптимальный план (построчно):")
    print(x_vec)
    print(f"Минимальная стоимость F* = {total_cost(C, final_sol.x)}")


if __name__ == "__main__":
    solve_and_print(SUPPLY, DEMAND, COSTS)

