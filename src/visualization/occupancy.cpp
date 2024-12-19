#include "pytimeloop/visualization/occupancy.h"

#include <cassert>

#include <isl/map.h>
#include <isl/set.h>
#include <barvinok/isl.h>


isl_stat SetScanner(isl_point*, void*);


struct SetScannerData
{
  int tensor_id;
  int n_dims;
  int n_dim_max;
  isl_pw_qpolynomial* p_occ;
  std::vector<long>* p_max_temporal_steps;
  std::vector<std::vector<int>>* p_data;
};


OccupancyMatrix ComputeOccupancyMatrix(
  const std::map<int, isl_pw_qpolynomial*>& occupancy
)
{
  int n_dim_max = 0;
  for (auto [tensor_id, p_occ] : occupancy)
  {
    isl_set* p_dom = isl_pw_qpolynomial_domain(
      isl_pw_qpolynomial_copy(p_occ)
    );
    int n_dim = isl_set_dim(p_dom, isl_dim_set);
    n_dim_max = n_dim_max > n_dim ? n_dim_max : n_dim;
  }

  std::vector<long> max_temporal_steps(n_dim_max, 0);
  for (auto [tensor_id, p_occ] : occupancy)
  {
    p_occ = isl_pw_qpolynomial_copy(p_occ);
    isl_set* p_dom = isl_pw_qpolynomial_domain(
      isl_pw_qpolynomial_copy(p_occ)
    );
    int n_dim = isl_set_dim(isl_set_copy(p_dom), isl_dim_set);
    if (n_dim == n_dim_max)
    {
      isl_set* p_max = isl_set_lexmax(p_dom);
      assert(isl_set_is_singleton(p_max));
      isl_point* p_max_point = isl_set_sample_point(p_max);
      for (int i = 0; i < n_dim_max; ++i)
      {
        isl_val* p_coord_val = isl_point_get_coordinate_val(p_max_point,
                                                            isl_dim_set,
                                                            i);
        max_temporal_steps.at(i) = std::max(max_temporal_steps.at(i),
                                            isl_val_get_num_si(p_coord_val)+1);
      }
      isl_point_free(p_max_point);
    }
  }

  auto occupancy_matrix = OccupancyMatrix{};
  occupancy_matrix.max_temporal_steps = 1;
  for (auto max_steps : max_temporal_steps)
  {
    occupancy_matrix.max_temporal_steps *= max_steps;
  }

  // Order occupancy based on (n_dims, lexmin, tensor_id)
  std::map<std::tuple<int, long, int>, isl_pw_qpolynomial*> ordered_occupancy;
  for (auto [tensor_id, p_occ] : occupancy)
  {
    isl_set* p_dom = isl_pw_qpolynomial_domain(
      isl_pw_qpolynomial_copy(p_occ)
    );
    int n_dim = isl_set_dim(isl_set_copy(p_dom), isl_dim_set);

    isl_set* p_lexmin = isl_set_lexmin(p_dom);
    assert(isl_set_is_singleton(p_lexmin));
    isl_point* p_min_point = isl_set_sample_point(p_lexmin);
    long start_temporal_step = 0;
    long factor = 1;
    for (int i = n_dim-1; i >= 0; --i)
    {
      isl_val* p_coord_val = isl_point_get_coordinate_val(p_min_point,
                                                          isl_dim_set,
                                                          i);
      long coord = isl_val_get_num_si(p_coord_val);
      start_temporal_step += coord*factor;
      factor *= max_temporal_steps.at(i);
      isl_val_free(p_coord_val);
    }
    isl_point_free(p_min_point);

    ordered_occupancy[std::make_tuple(n_dim, start_temporal_step, tensor_id)] = p_occ;
  }

  for (auto [n_dims_start_step_tensor_id, p_occ] : ordered_occupancy)
  {
    int n_dims = std::get<0>(n_dims_start_step_tensor_id);
    int tensor_id = std::get<2>(n_dims_start_step_tensor_id);

    SetScannerData scanner_data;
    scanner_data.tensor_id = tensor_id;
    scanner_data.n_dims = n_dims;
    scanner_data.n_dim_max = n_dim_max;
    scanner_data.p_occ = isl_pw_qpolynomial_copy(p_occ);
    scanner_data.p_max_temporal_steps = &max_temporal_steps;
    scanner_data.p_data = &occupancy_matrix.data;

    isl_set* p_dom = isl_pw_qpolynomial_domain(isl_pw_qpolynomial_copy(p_occ));
    isl_set_foreach_point(p_dom, SetScanner, &scanner_data);
  }

  size_t max_occupancy = 0;
  for (auto& column : occupancy_matrix.data)
  {
    max_occupancy = std::max(max_occupancy, column.size());
  }
  occupancy_matrix.max_occupancy = max_occupancy;

  for (auto& column : occupancy_matrix.data)
  {
    column.resize(max_occupancy, -1);
  }

  return occupancy_matrix;
}


isl_stat SetScanner(isl_point* p_point, void* p_voided_scanner_data)
{
  SetScannerData* p_scanner_data = 
    static_cast<SetScannerData*>(p_voided_scanner_data);
  isl_pw_qpolynomial* p_occ = p_scanner_data->p_occ;
  isl_val* p_tile_size_val = isl_pw_qpolynomial_eval(
    isl_pw_qpolynomial_copy(p_occ),
    isl_point_copy(p_point)
  );
  long tile_size = isl_val_get_num_si(p_tile_size_val);

  int n_dim_max = p_scanner_data->n_dim_max;
  int factor = 1;
  for (int i = n_dim_max-1; i >= p_scanner_data->n_dims; --i)
  {
    factor *= p_scanner_data->p_max_temporal_steps->at(i);
  }
  long temporal_steps = factor;
  long start_temporal_step = 0;
  for (int i = p_scanner_data->n_dims-1; i >= 0; --i)
  {
    start_temporal_step += isl_val_get_num_si(isl_point_get_coordinate_val(
      isl_point_copy(p_point),
      isl_dim_set,
      i
    ))*factor;
    factor *= p_scanner_data->p_max_temporal_steps->at(i);
  }
  long end_temporal_step = start_temporal_step + temporal_steps;

  std::vector<std::vector<int>>& data = *p_scanner_data->p_data;
  for (long temporal_step = start_temporal_step;
       temporal_step < end_temporal_step;
       ++temporal_step)
  {
    data.resize(std::max((long) data.size(), temporal_step+1));
    std::vector<int>& column = data.at(temporal_step);
    for (int i = 0; i < tile_size; ++i)
    {
      column.push_back(p_scanner_data->tensor_id);
    }
  }
  return isl_stat_ok;
}
