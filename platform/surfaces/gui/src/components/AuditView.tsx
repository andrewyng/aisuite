import { AuditTab } from "./ManageModal";
import { Icon } from "./Icon";

export function AuditView() {
  return (
    <div className="main">
      <div className="sa-view-head">
        <div className="sa-view-title">
          <Icon name="audit" size={19} className="mark" /> Audit
        </div>
        <div className="sa-view-sub">
          Filterable history of connector and browser tool activity.
        </div>
      </div>
      <div className="main-scroll">
        <div className="page-panel">
          <AuditTab />
        </div>
      </div>
    </div>
  );
}
