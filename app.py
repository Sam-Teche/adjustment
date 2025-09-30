from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import re

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# ========== Helpers ==========

def create_distance_matrix(df):
    """Diagonal weight matrix from distances."""
    n = len(df)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        distance_matrix[i, i] = df['Distances_Km'].iloc[i]
    return distance_matrix


def modify_equation(formula):
    """Standardize equation by tagging variables with V indices."""
    parts = formula.replace('-', ' -').replace('+', ' +').split()
    result = []

    for part in parts:
        if part in ['+', '-']:
            result.append(part)
            continue

        operator = ''
        if part.startswith(('+', '-')):
            operator, part = part[0], part[1:]
            result.append(operator)

        if re.match(r'[hH]\d+', part):  # h1, h2, ...
            idx = re.match(r'[hH](\d+)', part).group(1)
            result.append(f'V{idx} + {part}')
        else:
            result.append(part)

    return ' '.join(result)


def extract_coefficients(modified_eqn):
    """Extract coefficients like V1, V2 = ±1."""
    coeffs = {}
    for sign, idx in re.findall(r'([+-]?)\s*V(\d+)', modified_eqn):
        coeffs[int(idx)] = '+1' if (sign == '+' or sign == '') else '-1'
    return coeffs


def convert_coeff_str(val):
    """Convert string coeff to numeric."""
    if val in ['0', 0]:
        return 0
    return 1 if str(val).startswith('+') else -1


def transpose_matrix(matrix):
    """Transpose coefficient matrix."""
    names = [row[0] for row in matrix]
    num_cols = len(matrix[0]) - 1
    transposed = [["Parameter"] + names]
    for j in range(1, num_cols + 1):
        transposed.append([f"V{j}"] + [matrix[i][j] for i in range(len(matrix))])
    return transposed


def multiply_coeff_with_distances(coefficient_table, distance_matrix):
    """B × P (keep numeric)."""
    result_table = []
    for row in coefficient_table:
        name, coeffs = row[0], row[1:]
        nums = [convert_coeff_str(c) for c in coeffs]
        weighted = [coeff * distance_matrix[i, i] for i, coeff in enumerate(nums)]
        result_table.append([name] + weighted)
    return result_table


def multiply_with_transpose(result_table):
    """(BP) × (BP)ᵀ."""
    names = [row[0] for row in result_table]
    numeric_matrix = np.array([row[1:] for row in result_table], dtype=float)
    product = numeric_matrix @ numeric_matrix.T
    return [[names[i]] + list(product[i]) for i in range(len(names))]


def inverse_matrix(product_table):
    """Inverse of product table."""
    names = [row[0] for row in product_table]
    numeric = np.array([row[1:] for row in product_table], dtype=float)
    inv = np.linalg.inv(numeric)
    return [[names[i]] + list(inv[i]) for i in range(len(names))]


def multiply_inv_with_w(inverse_table, w_values):
    """K = -M⁻¹ × W."""
    inv = np.array([row[1:] for row in inverse_table], dtype=float)
    W = np.array(w_values, dtype=float)
    K = -(inv @ W)
    return [["K" + str(i+1), K[i]] for i in range(len(K))]


def calc_w_values(formulas, df, benchmarks):
    """Compute misclosures W."""
    W = []
    for formula in formulas:
        left, right = formula.split('=')
        left_val, right_val = 0, 0

        # Parse sides
        for sign, vtype, idx in re.findall(r'([+-]?)([hHbB])(\d+)', left):
            mult = 1 if (sign == '+' or sign == '') else -1
            idx = int(idx)
            if vtype.lower() == 'h':
                left_val += mult * df['h_obs'].iloc[idx-1]
            else:
                left_val += mult * benchmarks[idx-1]

        for sign, vtype, idx in re.findall(r'([+-]?)([hHbB])(\d+)', right):
            mult = 1 if (sign == '+' or sign == '') else -1
            idx = int(idx)
            if vtype.lower() == 'h':
                right_val += mult * df['h_obs'].iloc[idx-1]
            else:
                right_val += mult * benchmarks[idx-1]

        W.append(-(right_val - left_val))
    return W


def calc_residuals(distance_matrix, coeff_table, K):
    """Residuals: V = -P Bᵀ M⁻¹ W (simplified)."""
    P = np.diag(np.diag(distance_matrix))
    B = np.array([[convert_coeff_str(c) for c in row[1:]] for row in coeff_table], dtype=float)
    M = (B @ P) @ (B @ P).T
    Minv = np.linalg.inv(M)
    W = -np.array([float(val[1]) for val in K])
    V = -P @ B.T @ Minv @ W
    return [["Residual " + str(i+1), V[i]] for i in range(len(V))]


def adjusted_heights(df, residual):
    """Compute adjusted heights h + V."""
    res_vals = [float(r[1]) for r in residual]
    return [[f"Adjusted Height {i+1}", df['h_obs'].iloc[i] + res_vals[i]]
            for i in range(len(df))]


def format_output(table, decimals=4):
    """Format values for JSON output."""
    formatted = []
    for row in table:
        formatted_row = [row[0]]
        for val in row[1:]:
            if abs(val) < 1e-10:
                formatted_row.append("0")
            else:
                formatted_row.append(f"{val:+.{decimals}f}")
        formatted.append(formatted_row)
    return formatted


# ========== API Routes ==========

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "message": "API is running"}), 200


@app.route('/api/adjustment', methods=['POST'])
def adjustment():
    try:
        data = request.get_json()
        benchmarks, observations, num_params, formulas = (
            data['benchmarks'], data['observations'],
            data['num_parameters'], data['formulas']
        )

        df = pd.DataFrame(observations)
        df['Distances_Km'] = df['Distances_Km'].astype(float)
        df['h_obs'] = df['h_obs'].astype(float)

        n, r = len(df), len(df) - num_params
        if r <= 0:
            return jsonify({"error": "No condition equations required"}), 400

        distance_matrix = create_distance_matrix(df)

        coeff_table, modified_eqns = [], []
        for i, f in enumerate(formulas):
            meqn = modify_equation(f)
            modified_eqns.append(meqn)
            coeffs = extract_coefficients(meqn)
            row = [f"Eqn {i+1}"] + [coeffs.get(j, '0') for j in range(1, n+1)]
            coeff_table.append(row)

        result_table = multiply_coeff_with_distances(coeff_table, distance_matrix)
        W = calc_w_values(formulas, df, benchmarks)
        for i, row in enumerate(result_table):
            row.append(W[i])

        product_table = multiply_with_transpose(result_table)
        inv_table = inverse_matrix(product_table)
        K = multiply_inv_with_w(inv_table, W)
        residual = calc_residuals(distance_matrix, coeff_table, K)
        h_adj = adjusted_heights(df, residual)

        return jsonify({
            "num_observations": n,
            "num_parameters": num_params,
            "num_condition_eqns": r,
            "modified_equations": modified_eqns,
            "coefficient_table": coeff_table,
            "result_table": format_output(result_table),
            "product_table": format_output(product_table),
            "inverse_table": format_output(inv_table, 8),
            "k_values": format_output(K, 8),
            "residuals": format_output(residual, 8),
            "adjusted_heights": format_output(h_adj, 4)
        }), 200

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
