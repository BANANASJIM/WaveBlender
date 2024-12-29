/** (c) 2024 Kangrui Xue
 *
 * \file Object.cpp
 */

#include "Shaders.h"


void Object::_readAnimation()
{
	int lookahead = _N_samples - 1;
	std::string line;

	if (_step == 0) { std::getline(_animFileStream, line); }  // First line is metadata

	while (_animFileStream.good() && !_animFileStream.eof() && _t2 <= (_step + lookahead) * _dt + _ts)
	{
		std::getline(_animFileStream, line);
		_t1 = _t2; _translation1 = _translation2; _rotation1 = _rotation2;
		
		std::istringstream is(line);
		is >> _t2 >> _translation2[0] >> _translation2[1] >> _translation2[2] 
			>> _rotation2[0] >> _rotation2[1] >> _rotation2[2] >> _rotation2[3];
	}
	if (_step == 0) { _translation1 = _translation2; _rotation1 = _rotation2; }

	std::cout << _t1 << ", " << _t2 << std::endl;

	// TODO: better threshold
	if (_step > 0 && (_translation2 - _translation1).norm() < 1e-6 && (_rotation2 - _rotation1).norm() < 1e-6)
	{
		_changed = false; return;
	}
	// Scale Rot Trans
	Eigen::Quaterniond quat1(_rotation1[0], _rotation1[1], _rotation1[2], _rotation1[3]);
	Eigen::Quaterniond quat2(_rotation2[0], _rotation2[1], _rotation2[2], _rotation2[3]);

	double alpha = ((_step + lookahead) * _dt + _ts - _t1) / (_t2 - _t1);
	Eigen::Quaternion<REAL> quat = quat1.slerp(alpha, quat2).cast<REAL>();
	Eigen::Vector3<REAL> translation = ((1. - alpha) * _translation1 + alpha * _translation2).cast<REAL>();

	_V1 = _V2;
	for (int r = 0; r < _V2.rows(); r++)
	{
		_V2.row(r) = quat * _V0.row(r);
		_V2.row(r) += translation;
	}
	_changed = true;
}

/**
 * \private For each query position, computes the closest point on this Object's surface mesh (must be called after _tree.init())
 * \param[out] I  (_N_query) vector of triangle indices for each closest point
 * \param[in]  B  (_N_query, 3) matrix of query positions
 * \param[in]  V  (_N_verts, 3) matrix of mesh vertex positions (assumes connectivity _F does not change)
 */
void Object::_closestPoint(Eigen::VectorXi& I, const Eigen::MatrixX<REAL>& B, const Eigen::MatrixX<REAL>& V)
{
	Eigen::VectorX<REAL> sqrD; Eigen::MatrixX<REAL> P;
	_tree.squared_distance(V, _F, B, sqrD, I, P);
}

/**
 * \private Closest point query with additional barycentric weight computation
 * \param[out] W  (_Nquery, 3) matrix of barycentric weights for each closest point
 */
void Object::_closestPoint(Eigen::VectorXi& I, Eigen::MatrixX<REAL>& W, const Eigen::MatrixX<REAL>& B, const Eigen::MatrixX<REAL>& V)
{
	Eigen::VectorX<REAL> sqrD; Eigen::MatrixX<REAL> P;
	_tree.squared_distance(V, _F, B, sqrD, I, P);

	Eigen::MatrixX<REAL> Tri1(_N_points, 3);
	Eigen::MatrixX<REAL> Tri2(_N_points, 3);
	Eigen::MatrixX<REAL> Tri3(_N_points, 3);
	for (int bid = 0; bid < _N_points; bid++)
	{
		Tri1.row(bid) = V.row(_F(I[bid], 0));
		Tri2.row(bid) = V.row(_F(I[bid], 1));
		Tri3.row(bid) = V.row(_F(I[bid], 2));
	}
	igl::barycentric_coordinates(P, Tri1, Tri2, Tri3, W);
}