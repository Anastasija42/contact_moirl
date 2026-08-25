#include <crocoddyl/core/actuation-base.hpp>
#include <crocoddyl/multibody/states/multibody.hpp>
#include <crocoddyl/core/utils/exception.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/contact-dynamics.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/frames-derivatives.hpp>
#include <pinocchio/algorithm/rnea-second-order-derivatives.hpp>
#include <pinocchio/container/aligned-vector.hpp>
#include <eigenpy/eigenpy.hpp> 
#include <boost/python.hpp>
#include <memory>

namespace crocoddyl {

template <typename Scalar>
struct ActuationDataFrictionTpl : public ActuationDataAbstractTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef MathBaseTpl<Scalar> MathBase;
  
  std::shared_ptr<pinocchio::DataTpl<Scalar> > pin_data;
  typename MathBase::Matrix6xs J; 

  template <template <typename ModelScalar> class Model>
  ActuationDataFrictionTpl(Model<Scalar>* const model,
                           const pinocchio::ModelTpl<Scalar>& pin_model)
      : ActuationDataAbstractTpl<Scalar>(model),
        J(6, model->get_state()->get_nv()) {
    pin_data = std::make_shared<pinocchio::DataTpl<Scalar> >(pin_model);
    J.setZero();
  }
};

template <typename _Scalar>
class ActuationModelFrictionTpl : public ActuationModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef _Scalar Scalar;
  typedef ActuationModelAbstractTpl<Scalar>  Base;
  typedef ActuationDataAbstractTpl<Scalar>   ActuationDataAbstract;
  typedef ActuationDataFrictionTpl<Scalar>   Data;       
  typedef MathBaseTpl<Scalar>                MathBase;
  typedef typename MathBase::VectorXs        VectorXs;
  typedef typename MathBase::MatrixXs        MatrixXs;
  typedef StateMultibodyTpl<Scalar>          StateMultibody;
  typedef Eigen::Matrix<Scalar, 3, 3>        Matrix3s;

  ActuationModelFrictionTpl(std::shared_ptr<StateMultibody> state,
                            const std::size_t frame_id,
                            const Scalar mu,
                            const Scalar target_fn)
      : Base(state, state->get_nv()),
        frame_id_(frame_id),
        mu_(mu),
        target_fn_(target_fn),
        eps_(Scalar(1e-3)),
        R_surface_(Matrix3s::Identity()),
        use_press_(false),
        inject_normal_(true) {}


  ActuationModelFrictionTpl(std::shared_ptr<StateMultibody> state,
                            const std::size_t frame_id,
                            const Scalar mu,
                            const Scalar target_fn,
                            const Matrix3s& R_surface)
      : Base(state, state->get_nv()),
        frame_id_(frame_id),
        mu_(mu),
        target_fn_(target_fn),
        eps_(Scalar(1e-3)),
        R_surface_(R_surface),
        use_press_(true),
        inject_normal_(true) {}

  virtual ~ActuationModelFrictionTpl() {}

  virtual std::shared_ptr<crocoddyl::ActuationModelBase> cloneAsDouble() const override {
    throw std::runtime_error("ActuationModelFriction::cloneAsDouble() not supported");
  }

  virtual std::shared_ptr<crocoddyl::ActuationModelBase> cloneAsFloat() const override {
    throw std::runtime_error("ActuationModelFriction::cloneAsFloat() not supported");
  }

  virtual void calc(const std::shared_ptr<ActuationDataAbstractTpl<Scalar> >& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) {
    Data* d = static_cast<Data*>(data.get());
    auto state_mb = std::static_pointer_cast<StateMultibody>(this->state_);
    const pinocchio::ModelTpl<Scalar>& model = *state_mb->get_pinocchio();

    pinocchio::forwardKinematics(model, *d->pin_data,
                                 x.head(state_mb->get_nq()),
                                 x.tail(state_mb->get_nv()));
    pinocchio::updateFramePlacements(model, *d->pin_data);

    pinocchio::MotionTpl<Scalar> v_frame = pinocchio::getFrameVelocity(
        model, *d->pin_data, frame_id_, pinocchio::LOCAL);

    pinocchio::ForceTpl<Scalar> f_fric = pinocchio::ForceTpl<Scalar>::Zero();
    if (use_press_) {
      // Match crocoddyl ContactModel1D EXACTLY: it forms Jc = (Raxis*fJf_lin).row(2),
      // i.e. surface frame = Raxis*(LOCAL). So velocity into surface = R_surface*v_loc,
      // and the wrench maps back to LOCAL with R_surface^T. This lands the normal on
      // R_surface.row(2) (the contact's actual reaction axis) -> lambda ~ f_n,
      // transmission ~1.0 and subject-independent. See docs/press_force_transmission_problem.md
      Eigen::Matrix<Scalar, 3, 1> v_loc = v_frame.linear();
      Eigen::Matrix<Scalar, 3, 1> v_surf = R_surface_ * v_loc;
      Scalar vt = sqrt(v_surf.x()*v_surf.x() + v_surf.y()*v_surf.y() + eps_*eps_);
      Eigen::Matrix<Scalar, 3, 1> f_surf;
      f_surf.x() = -mu_ * target_fn_ * (v_surf.x() / vt);   // tangential friction
      f_surf.y() = -mu_ * target_fn_ * (v_surf.y() / vt);   // tangential friction
      // Normal press: injected only when inject_normal_ (demo/imposed). With
      // press_normal_dual the normal is left FREE (dual driven by press_force),
      // so f_surf.z=0 -> friction is purely surface-tangent, NO leak onto the
      // contact normal axis R_surface.row(2). See docs/rollout_force_dual_design.md.
      f_surf.z() = inject_normal_ ? -target_fn_ : Scalar(0);
      f_fric.linear() = R_surface_.transpose() * f_surf;    // back to LOCAL
    } else {
      Scalar vx     = v_frame.linear().x();
      Scalar vy     = v_frame.linear().y();
      Scalar v_norm = sqrt(vx*vx + vy*vy + eps_*eps_);
      f_fric.linear().x() = -mu_ * target_fn_ * (vx / v_norm);
      f_fric.linear().y() = -mu_ * target_fn_ * (vy / v_norm);
    }

    pinocchio::computeFrameJacobian(model, *d->pin_data,
                                    x.head(state_mb->get_nq()),
                                    frame_id_, pinocchio::LOCAL, d->J);

    data->tau = u + d->J.transpose() * f_fric.toVector();
  }

  virtual void calcDiff(const std::shared_ptr<ActuationDataAbstractTpl<Scalar> >& data,
                        const Eigen::Ref<const VectorXs>&,
                        const Eigen::Ref<const VectorXs>&) {
    data->dtau_du.setIdentity();
    data->dtau_dx.setZero();
  }

  // Fast C++ finite-difference of dtau_dx. The analytic dtau_dx of the friction
  // map (tau = u + J(q)^T f_fric(q,v)) is not provided; recover it by numdiff
  // IN C++ (no Python per-perturbation overhead). Stored in data->dtau_dx so the
  // contact-aware Eng/Tau feature residuals read it instead of their own Python
  // numdiff loop. Deliberately SEPARATE from calcDiff: the OCP dynamics keep
  // calling calcDiff (dtau_dx = 0), so the solve is byte-for-byte unchanged and
  // only the IRL feature gradient gets the speed-up.
  void calcTauJacobian(const std::shared_ptr<ActuationDataAbstractTpl<Scalar> >& data,
                       const Eigen::Ref<const VectorXs>& x,
                       const Eigen::Ref<const VectorXs>& u) {
    auto state_mb = std::static_pointer_cast<StateMultibody>(this->state_);
    const int ndx = static_cast<int>(state_mb->get_ndx());
    calc(data, x, u);
    VectorXs tau0 = data->tau;
    const Scalar fd_eps = Scalar(1e-6);
    VectorXs dx = VectorXs::Zero(ndx);
    VectorXs xp = VectorXs::Zero(x.size());
    for (int i = 0; i < ndx; ++i) {
      dx(i) = fd_eps;
      state_mb->integrate(x, dx, xp);
      calc(data, xp, u);
      data->dtau_dx.col(i) = (data->tau - tau0) / fd_eps;
      dx(i) = Scalar(0);
    }
    calc(data, x, u);            // restore data->tau/J to the unperturbed state
    data->dtau_du.setIdentity();
  }

  // ANALYTIC dtau_dx for tau = u + J^T f_fric(q,v)  (Phase 1, replaces the numdiff):
  //   dtau_dv = Jlin^T (R^T D R) Jlin
  //   dtau_dq = Jlin^T (R^T D R) dvloc_dq  +  (dJ^T/dq) f_fric
  // D = df_surf/dv_surf (friction-cone softening); dvloc_dq from getFrameVelocityDerivatives;
  // the (dJ^T/dq) f term via RNEA(q,0,0,fext): rnea=g-J^T fext, so d(J^T f)/dq = dg/dq - d(rnea_fext)/dq.
  void calcDiff_analytic(const std::shared_ptr<ActuationDataAbstractTpl<Scalar> >& data,
                         const Eigen::Ref<const VectorXs>& x,
                         const Eigen::Ref<const VectorXs>& u) {
    Data* d = static_cast<Data*>(data.get());
    auto state_mb = std::static_pointer_cast<StateMultibody>(this->state_);
    const pinocchio::ModelTpl<Scalar>& model = *state_mb->get_pinocchio();
    const int nq = model.nq, nv = model.nv;
    VectorXs q = x.head(nq), v = x.tail(nv);
    data->dtau_du.setIdentity();
    data->dtau_dx.setZero();

    pinocchio::computeForwardKinematicsDerivatives(model, *d->pin_data, q, v, VectorXs::Zero(nv));
    pinocchio::updateFramePlacements(model, *d->pin_data);
    pinocchio::MotionTpl<Scalar> v_frame =
        pinocchio::getFrameVelocity(model, *d->pin_data, frame_id_, pinocchio::LOCAL);
    Eigen::Matrix<Scalar,3,1> v_loc = v_frame.linear();
    Matrix3s R = use_press_ ? R_surface_ : Matrix3s::Identity();
    Eigen::Matrix<Scalar,3,1> v_surf = R * v_loc;
    Scalar vt = std::sqrt(v_surf.x()*v_surf.x() + v_surf.y()*v_surf.y() + eps_*eps_);
    Scalar c = mu_ * target_fn_, vt3 = vt*vt*vt;
    Matrix3s Dc = Matrix3s::Zero();
    Dc(0,0) = -c*(v_surf.y()*v_surf.y() + eps_*eps_)/vt3;
    Dc(0,1) =  c*v_surf.x()*v_surf.y()/vt3;
    Dc(1,0) =  c*v_surf.x()*v_surf.y()/vt3;
    Dc(1,1) = -c*(v_surf.x()*v_surf.x() + eps_*eps_)/vt3;
    Matrix3s dfl_dvloc = R.transpose() * Dc * R;

    typename pinocchio::DataTpl<Scalar>::Matrix6x J6(6, nv); J6.setZero();
    pinocchio::getFrameJacobian(model, *d->pin_data, frame_id_, pinocchio::LOCAL, J6);
    Eigen::Matrix<Scalar,3,Eigen::Dynamic> Jlin = J6.template topRows<3>();
    typename pinocchio::DataTpl<Scalar>::Matrix6x vdq(6, nv), vdv(6, nv);
    vdq.setZero(); vdv.setZero();
    pinocchio::getFrameVelocityDerivatives(model, *d->pin_data, frame_id_, pinocchio::LOCAL, vdq, vdv);
    Eigen::Matrix<Scalar,3,Eigen::Dynamic> dvloc_dq = vdq.template topRows<3>();

    MatrixXs dtau_dv  = Jlin.transpose() * (dfl_dvloc * Jlin);
    MatrixXs dtau_dq1 = Jlin.transpose() * (dfl_dvloc * dvloc_dq);   // J^T df/dq  (Part 1)

    Eigen::Matrix<Scalar,3,1> f_surf;
    f_surf.x() = -c * v_surf.x()/vt;
    f_surf.y() = -c * v_surf.y()/vt;
    // Match calc EXACTLY: the normal is injected ONLY in the use_press_ branch.
    // With use_press_=False (the default recovery actuation) the friction is pure
    // tangential and the normal is the free contact dual, so f_z = 0 here.
    f_surf.z() = (use_press_ && inject_normal_) ? -target_fn_ : Scalar(0);
    Eigen::Matrix<Scalar,3,1> f_lin = R.transpose() * f_surf;
    pinocchio::ForceTpl<Scalar> f_fric(f_lin, Eigen::Matrix<Scalar,3,1>::Zero());
    pinocchio::container::aligned_vector<pinocchio::ForceTpl<Scalar> >
        fext(model.njoints, pinocchio::ForceTpl<Scalar>::Zero());
    pinocchio::JointIndex jid = model.frames[frame_id_].parentJoint;
    fext[jid] = model.frames[frame_id_].placement.act(f_fric);
    VectorXs z = VectorXs::Zero(nv);
    MatrixXs dq0(nv,nv), dv0(nv,nv), da0(nv,nv), dqf(nv,nv);
    pinocchio::computeRNEADerivatives(model, *d->pin_data, q, z, z, dq0, dv0, da0);
    pinocchio::computeRNEADerivatives(model, *d->pin_data, q, z, z, fext, dqf, dv0, da0);
    MatrixXs dtau_dq2 = dq0 - dqf;                                   // (dJ^T/dq) f  (Part 2)

    data->dtau_dx.leftCols(nv)  = dtau_dq1 + dtau_dq2;               // J^T df/dq + (dJ^T/dq) f
    data->dtau_dx.rightCols(nv) = dtau_dv;
  }

  // Analytic dg/d(q,v,a) for the JTC torque-rate g = dtau_dq.v + dtau_dv.a, via the
  // second-order RNEA (kills the ~2*nv computeRNEADerivatives of the JTC numdiff).
  //   dg_dq[m,j] = sum_i d2tau_dqdq[m,i,j] v_i + d2tau_dqdv[m,j,i] a_i
  //   dg_dv[m,j] = sum_i d2tau_dqdv[m,i,j] v_i + d2tau_dvdv[m,i,j] a_i + dtau_dq[m,j]
  //   dg_da[m,j] = sum_i d2tau_dadq[m,j,i] v_i + dtau_dv[m,j]
  boost::python::tuple jtc_dg(const std::shared_ptr<ActuationDataAbstractTpl<Scalar> >& data,
                             const Eigen::Ref<const VectorXs>& x,
                             const Eigen::Ref<const VectorXs>& a) {
    Data* d = static_cast<Data*>(data.get());
    auto state_mb = std::static_pointer_cast<StateMultibody>(this->state_);
    const pinocchio::ModelTpl<Scalar>& model = *state_mb->get_pinocchio();
    const int nq = model.nq, nv = model.nv;
    VectorXs q = x.head(nq), vv = x.tail(nv);
    pinocchio::computeRNEADerivatives(model, *d->pin_data, q, vv, a);
    MatrixXs dtau_dq = d->pin_data->dtau_dq;
    MatrixXs dtau_dv = d->pin_data->dtau_dv;
    pinocchio::ComputeRNEASecondOrderDerivatives(model, *d->pin_data, q, vv, a);
    const Scalar* Tqq = d->pin_data->d2tau_dqdq.data();
    const Scalar* Tvv = d->pin_data->d2tau_dvdv.data();
    const Scalar* Tqv = d->pin_data->d2tau_dqdv.data();
    const Scalar* Taq = d->pin_data->d2tau_dadq.data();
    auto IDX = [nv](int m, int i, int j){ return m + nv*i + nv*nv*j; };  // col-major (m,i,j)
    MatrixXs dg_dq = MatrixXs::Zero(nv,nv), dg_dv = MatrixXs::Zero(nv,nv), dg_da = MatrixXs::Zero(nv,nv);
    for (int m=0;m<nv;++m) for (int j=0;j<nv;++j) {
      Scalar sq=0, sv=0, sa=0;
      for (int i=0;i<nv;++i) {
        sq += Tqq[IDX(m,i,j)]*vv(i) + Tqv[IDX(m,j,i)]*a(i);
        sv += Tqv[IDX(m,i,j)]*vv(i) + Tvv[IDX(m,i,j)]*a(i);
        sa += Taq[IDX(m,j,i)]*vv(i);
      }
      dg_dq(m,j)=sq;  dg_dv(m,j)=sv+dtau_dq(m,j);  dg_da(m,j)=sa+dtau_dv(m,j);
    }
    return boost::python::make_tuple(dg_dq, dg_dv, dg_da);
  }

  // FULL analytic JTC calcDiff in C++ (no Python per-node overhead). Assembles the
  // KKT sensitivity da/dx + 2nd-order RNEA dg + friction tau-path (see _jtc_analytic
  // in utils_model_residuals.py for the validated reference). Returns (Rx, Ru).
  boost::python::tuple jtc_analytic_calcDiff(
      const std::shared_ptr<ActuationDataAbstractTpl<Scalar> >& data,
      const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u,
      int frame_id, const Eigen::Ref<const Matrix3s>& R_axis, Scalar eps) {
    Data* d = static_cast<Data*>(data.get());
    auto state_mb = std::static_pointer_cast<StateMultibody>(this->state_);
    const pinocchio::ModelTpl<Scalar>& model = *state_mb->get_pinocchio();
    pinocchio::DataTpl<Scalar>& pd = *d->pin_data;
    const int nq = model.nq, nv = model.nv;
    VectorXs q = x.head(nq), v = x.tail(nv), zero = VectorXs::Zero(nv);

    calc(data, x, u);                       // tau = u + Jc^T f_fric
    VectorXs tau = d->tau;

    pinocchio::computeAllTerms(model, pd, q, v);
    MatrixXs M = pd.M;
    M.template triangularView<Eigen::StrictlyLower>() =
        M.transpose().template triangularView<Eigen::StrictlyLower>();
    pinocchio::updateFramePlacements(model, pd);
    typename pinocchio::DataTpl<Scalar>::Matrix6x J6(6, nv); J6.setZero();
    pinocchio::getFrameJacobian(model, pd, frame_id, pinocchio::LOCAL, J6);
    Eigen::Matrix<Scalar,1,Eigen::Dynamic> Jc =
        (R_axis.transpose() * J6.template topRows<3>()).row(2);      // 1 x nv

    auto gamma_at = [&](const VectorXs& qq, const VectorXs& vv)->Scalar {
      pinocchio::forwardKinematics(model, pd, qq, vv, zero);
      pinocchio::updateFramePlacements(model, pd);
      Eigen::Matrix<Scalar,3,1> drift =
          pinocchio::getFrameClassicalAcceleration(model, pd, frame_id, pinocchio::LOCAL).linear();
      return (R_axis.transpose() * drift)(2);
    };
    Scalar g0 = gamma_at(q, v);

    MatrixXs Jc_m = Jc;  VectorXs gvec(1); gvec(0) = g0;
    pinocchio::forwardDynamics(model, pd, q, v, tau, Jc_m, gvec);
    VectorXs a0 = pd.ddq;
    MatrixXs Minv = M.inverse();
    Scalar K = (Jc * Minv * Jc.transpose())(0,0);
    VectorXs a_free = pinocchio::aba(model, pd, q, v, tau);
    Scalar lam = -((Jc * a_free)(0) + g0) / K;

    pinocchio::computeRNEADerivatives(model, pd, q, v, a0);
    MatrixXs dtau_dq = pd.dtau_dq, dtau_dv = pd.dtau_dv;

    VectorXs JcTl0 = Jc.transpose() * lam;  Scalar Jca0 = (Jc * a0)(0);
    MatrixXs dJcTl(nv, nv);  MatrixXs dJca(1, nv), dgq(1, nv), dgv(1, nv);
    for (int i = 0; i < nv; ++i) {
      VectorXs dq_i = VectorXs::Zero(nv); dq_i(i) = eps;
      VectorXs qp(nq); pinocchio::integrate(model, q, dq_i, qp);
      typename pinocchio::DataTpl<Scalar>::Matrix6x J6p(6, nv); J6p.setZero();
      pinocchio::computeFrameJacobian(model, pd, qp, frame_id, pinocchio::LOCAL, J6p);
      Eigen::Matrix<Scalar,1,Eigen::Dynamic> Jcp = (R_axis.transpose()*J6p.template topRows<3>()).row(2);
      dJcTl.col(i) = (Jcp.transpose()*lam - JcTl0) / eps;
      dJca(0,i)    = ((Jcp * a0)(0) - Jca0) / eps;
      dgq(0,i)     = (gamma_at(qp, v) - g0) / eps;
      VectorXs vp = v; vp(i) += eps;
      dgv(0,i)     = (gamma_at(q, vp) - g0) / eps;
    }

    MatrixXs KKT = MatrixXs::Zero(nv+1, nv+1);
    KKT.topLeftCorner(nv,nv) = M;
    KKT.topRightCorner(nv,1) = Jc.transpose();
    KKT.bottomLeftCorner(1,nv) = Jc;
    Eigen::FullPivLU<MatrixXs> kktlu(KKT);
    MatrixXs rhs_q(nv+1, nv); rhs_q.topRows(nv) = -dtau_dq + dJcTl; rhs_q.bottomRows(1) = -dJca - dgq;
    MatrixXs rhs_v(nv+1, nv); rhs_v.topRows(nv) = -dtau_dv;          rhs_v.bottomRows(1) = -dgv;
    MatrixXs da_dq_dir = kktlu.solve(rhs_q).topRows(nv);
    MatrixXs da_dv_dir = kktlu.solve(rhs_v).topRows(nv);

    MatrixXs da_dtau = Minv - (Minv * Jc.transpose() * Jc * Minv) / K;
    calcDiff_analytic(data, x, u);
    MatrixXs dtaudx = d->dtau_dx, dtaudu = d->dtau_du;
    MatrixXs da_dq = da_dq_dir + da_dtau * dtaudx.leftCols(nv);
    MatrixXs da_dv = da_dv_dir + da_dtau * dtaudx.rightCols(nv);
    MatrixXs da_du = da_dtau * dtaudu;

    pinocchio::computeRNEADerivatives(model, pd, q, v, a0);
    MatrixXs dtq = pd.dtau_dq, dtv = pd.dtau_dv;
    pinocchio::ComputeRNEASecondOrderDerivatives(model, pd, q, v, a0);
    const Scalar* Tqq = pd.d2tau_dqdq.data();  const Scalar* Tvv = pd.d2tau_dvdv.data();
    const Scalar* Tqv = pd.d2tau_dqdv.data();  const Scalar* Taq = pd.d2tau_dadq.data();
    auto IDX = [nv](int m,int i,int j){ return m + nv*i + nv*nv*j; };
    MatrixXs dg_dq = MatrixXs::Zero(nv,nv), dg_dv = MatrixXs::Zero(nv,nv), dg_da = MatrixXs::Zero(nv,nv);
    for (int m=0;m<nv;++m) for (int j=0;j<nv;++j) {
      Scalar sq=0,sv=0,sa=0;
      for (int i=0;i<nv;++i){
        sq += Tqq[IDX(m,i,j)]*v(i) + Tqv[IDX(m,j,i)]*a0(i);
        sv += Tqv[IDX(m,i,j)]*v(i) + Tvv[IDX(m,i,j)]*a0(i);
        sa += Taq[IDX(m,j,i)]*v(i);
      }
      dg_dq(m,j)=sq; dg_dv(m,j)=sv+dtq(m,j); dg_da(m,j)=sa+dtv(m,j);
    }

    MatrixXs Rx(nv, 2*nv);
    Rx.leftCols(nv)  = dg_dq + dg_da*da_dq;
    Rx.rightCols(nv) = dg_dv + dg_da*da_dv;
    MatrixXs Ru = dg_da * da_du;
    return boost::python::make_tuple(Rx, Ru);
  }

  virtual void commands(const std::shared_ptr<ActuationDataAbstractTpl<Scalar> >& data,
                        const Eigen::Ref<const VectorXs>&,
                        const Eigen::Ref<const VectorXs>& tau) {
    data->u = tau; 
  }

  virtual std::shared_ptr<ActuationDataAbstractTpl<Scalar> > createData() {
    auto state_mb = std::static_pointer_cast<StateMultibody>(this->state_);
    return std::make_shared<Data>(this, *state_mb->get_pinocchio());
  }

  // Mutable normal force -> Option B (docs/rollout_force_dual_design.md §3.2):
  // after each solve the friction f_n is refreshed to the contact DUAL so the
  // tangential friction mu*f_n and the normal share ONE consistent f_n.
  Scalar get_target_fn() const { return target_fn_; }
  void   set_target_fn(const Scalar fn) { target_fn_ = fn; }

  // Surface-frame friction WITHOUT the normal press (press_normal_dual): set
  // inject_normal=False so the friction stays purely tangential (no leak onto
  // the normal axis) while the contact dual carries the (free) normal force.
  bool get_inject_normal() const { return inject_normal_; }
  void set_inject_normal(const bool v) { inject_normal_ = v; }

 private:
  pinocchio::FrameIndex frame_id_;
  Scalar mu_;
  Scalar target_fn_;
  Scalar eps_;
  Matrix3s R_surface_;   // surface frame (col 2 = contact normal); Identity for 4-arg
  bool use_press_;       // 5-arg path: surface-frame friction + normal press
  bool inject_normal_;   // 5-arg: also inject the normal press (false = friction only)
};

}  

namespace bp = boost::python;
using namespace crocoddyl;

// ---- Contact-aware JTC residual numdiff Jacobian, entirely in C++ ----
// r(x,u) = dtau_dq*v + dtau_dv*a, a = contact-constrained acceleration (or free aba if
// frame_id<0). Byte-identical to the Python _r_fd numdiff, but the ~(ndx+nu) perturbation
// loop and its pinocchio calls (forwardDynamics + computeRNEADerivatives) run natively.
static Eigen::VectorXd _jtc_r(const pinocchio::Model& model, pinocchio::Data& pd,
                              const std::shared_ptr<ActuationModelFrictionTpl<double>>& act,
                              const std::shared_ptr<ActuationDataFrictionTpl<double>>& ad,
                              const Eigen::VectorXd& x, const Eigen::VectorXd& u,
                              int frame_id, const Eigen::Matrix3d& R_axis) {
  const int nq = model.nq, nv = model.nv;
  act->calc(ad, x, u);
  Eigen::VectorXd tau = ad->tau;
  Eigen::VectorXd q = x.head(nq);
  Eigen::VectorXd v = x.segment(nq, nv);
  Eigen::VectorXd a;
  if (frame_id < 0) {
    pinocchio::aba(model, pd, q, v, tau);
    a = pd.ddq;
  } else {
    const pinocchio::FrameIndex fid = (pinocchio::FrameIndex)frame_id;
    pinocchio::computeAllTerms(model, pd, q, v);
    pinocchio::Data::Matrix6x J6(6, nv); J6.setZero();
    pinocchio::getFrameJacobian(model, pd, fid, pinocchio::LOCAL, J6);
    Eigen::MatrixXd Jc = (R_axis.transpose() * J6.topRows(3)).row(2);   // 1 x nv
    pinocchio::forwardKinematics(model, pd, q, v, Eigen::VectorXd::Zero(nv));
    Eigen::Vector3d drift =
        pinocchio::getFrameClassicalAcceleration(model, pd, fid, pinocchio::LOCAL).linear();
    Eigen::VectorXd gamma(1);
    gamma(0) = (R_axis.transpose() * drift)(2);
    pinocchio::forwardDynamics(model, pd, q, v, tau, Jc, gamma, 0.0);
    a = pd.ddq;
  }
  pinocchio::computeRNEADerivatives(model, pd, q, v, a);
  return pd.dtau_dq * v + pd.dtau_dv * a;
}

boost::python::tuple jtc_contact_calcDiff(
    std::shared_ptr<ActuationModelFrictionTpl<double>> act,
    std::shared_ptr<ActuationDataFrictionTpl<double>> ad,
    const Eigen::VectorXd& x, const Eigen::VectorXd& u,
    int frame_id, const Eigen::Matrix3d& R_axis, double eps) {
  auto state = std::static_pointer_cast<StateMultibodyTpl<double>>(act->get_state());
  const pinocchio::Model& model = *state->get_pinocchio();
  pinocchio::Data pd(model);
  const int nv = model.nv, ndx = (int)state->get_ndx(), nu = (int)u.size();
  Eigen::VectorXd r0 = _jtc_r(model, pd, act, ad, x, u, frame_id, R_axis);
  Eigen::MatrixXd Rx(nv, ndx), Ru(nv, nu);
  Eigen::VectorXd dx = Eigen::VectorXd::Zero(ndx), xp(x.size());
  for (int i = 0; i < ndx; ++i) {
    dx(i) = eps;
    state->integrate(x, dx, xp);
    Rx.col(i) = (_jtc_r(model, pd, act, ad, xp, u, frame_id, R_axis) - r0) / eps;
    dx(i) = 0.0;
  }
  Eigen::VectorXd du = Eigen::VectorXd::Zero(nu);
  for (int i = 0; i < nu; ++i) {
    du(i) = eps;
    Ru.col(i) = (_jtc_r(model, pd, act, ad, x, u + du, frame_id, R_axis) - r0) / eps;
    du(i) = 0.0;
  }
  return boost::python::make_tuple(Rx, Ru);
}

BOOST_PYTHON_MODULE(friction_lib) {
  eigenpy::enableEigenPy();

  typedef ActuationModelFrictionTpl<double>  ActuationModelFriction;
  typedef ActuationDataFrictionTpl<double>   ActuationDataFriction;

  bp::class_<ActuationDataFriction,
             bp::bases<crocoddyl::ActuationDataAbstractTpl<double>>,
             std::shared_ptr<ActuationDataFriction>,
             boost::noncopyable>("ActuationDataFriction", bp::no_init)
    .def_readonly("J", &ActuationDataFriction::J);

  bp::class_<ActuationModelFriction,
             bp::bases<ActuationModelAbstractTpl<double>>,
             std::shared_ptr<ActuationModelFriction>,
             boost::noncopyable>(
      "ActuationModelFriction",
      bp::init<std::shared_ptr<StateMultibodyTpl<double>>,
               std::size_t,
               double,
               double>(
          bp::args("self", "state", "frame_id", "mu", "target_fn")))
    .def(bp::init<std::shared_ptr<StateMultibodyTpl<double>>,
                  std::size_t,
                  double,
                  double,
                  Eigen::Matrix<double, 3, 3>>(
          bp::args("self", "state", "frame_id", "mu", "target_fn", "R_surface")))
    .def("calc",       &ActuationModelFriction::calc,
         bp::args("self", "data", "x", "u"))
    .def("calcDiff",   &ActuationModelFriction::calcDiff,
         bp::args("self", "data", "x", "u"))
    .def("calcTauJacobian", &ActuationModelFriction::calcTauJacobian,
         bp::args("self", "data", "x", "u"))
    .def("calcDiff_analytic", &ActuationModelFriction::calcDiff_analytic,
         bp::args("self", "data", "x", "u"))
    .def("jtc_dg", &ActuationModelFriction::jtc_dg,
         bp::args("self", "data", "x", "a"))
    .def("jtc_analytic_calcDiff", &ActuationModelFriction::jtc_analytic_calcDiff,
         bp::args("self", "data", "x", "u", "frame_id", "R_axis", "eps"))
    .def("commands",   &ActuationModelFriction::commands,
         bp::args("self", "data", "x", "tau"))
    .def("createData", &ActuationModelFriction::createData)
    .add_property("nu", &ActuationModelFriction::get_nu)
    .add_property("target_fn", &ActuationModelFriction::get_target_fn,
                  &ActuationModelFriction::set_target_fn)
    .add_property("inject_normal", &ActuationModelFriction::get_inject_normal,
                  &ActuationModelFriction::set_inject_normal);

  bp::def("jtc_contact_calcDiff", &jtc_contact_calcDiff,
          bp::args("act", "act_data", "x", "u", "frame_id", "R_axis", "eps"));
}