export class Matrix3x3 {
    constructor(_00, _10, _20, _01, _11, _21, _02, _12, _22) {
        this._00 = _00;
        this._10 = _10;
        this._20 = _20;
        this._01 = _01;
        this._11 = _11;
        this._21 = _21;
        this._02 = _02;
        this._12 = _12;
        this._22 = _22;
    }

    static identity() {
        return new Matrix3x3(
            1, 0, 0,
            0, 1, 0,
            0, 0, 1
        )
    }

    static yFlipping() {
        return new Matrix3x3(
            1,  0, 0,
            0, -1, 0,
            0,  0, 1
        );
    }

    static scaling(vector2D) {
        return new Matrix3x3(
            vector2D.x,          0, 0,
                     0, vector2D.y, 0,
                     0,          0, 1
        );
    }

    static translation(vector2D) {
        return new Matrix3x3(
            1, 0, vector2D.x,
            0, 1, vector2D.y,
            0, 0,          1
        );
    }

    multiply(other) {
        return new Matrix3x3(
            this._00 * other._00 + this._10 * other._01 + this._20 * other._02,
            this._00 * other._10 + this._10 * other._11 + this._20 * other._12,
            this._00 * other._20 + this._10 * other._21 + this._20 * other._22,

            this._01 * other._00 + this._11 * other._01 + this._21 * other._02,
            this._01 * other._10 + this._11 * other._11 + this._21 * other._12,
            this._01 * other._20 + this._11 * other._21 + this._21 * other._22,

            this._02 * other._00 + this._12 * other._01 + this._22 * other._02,
            this._02 * other._10 + this._12 * other._11 + this._22 * other._12,
            this._02 * other._20 + this._12 * other._21 + this._22 * other._22
        );
    }

    inverse() {
        const det = this._00 * this._11 * this._22 + this._01 * this._12 * this._20 + this._02 * this._10 * this._21 - this._00 * this._12 * this._21 - this._01 * this._10 * this._22 - this._02 * this._11 * this._20;
        return new Matrix3x3(
            (this._11 * this._22 - this._12 * this._21) / det,
            (this._02 * this._21 - this._01 * this._22) / det,
            (this._01 * this._12 - this._02 * this._11) / det,
            (this._12 * this._20 - this._10 * this._22) / det,
            (this._00 * this._22 - this._02 * this._20) / det,
            (this._02 * this._10 - this._00 * this._12) / det,
            (this._10 * this._21 - this._11 * this._20) / det,
            (this._01 * this._20 - this._00 * this._21) / det,
            (this._00 * this._11 - this._01 * this._10) / det
        );
    }
}