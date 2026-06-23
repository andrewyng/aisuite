import { AuditTab } from "./ManageModal";
import { Icon } from "./Icon";

export function AuditView() {
  return (
    <div className="main page-view">
      <div className="page-col">
        <div className="sa-view-head">
          <div className="sa-view-heading">
            <div className="sa-view-title"><Icon name="audit" size={21} /> Audit</div>
            <div className="sa-view-sub">Filterable history of connector and browser tool activity.</div>
          </div>
        </div>
        <div className="main-scroll">
          <div className="page-panel">
            <AuditTab />
          </div>
        </div>
      </div>
    </div>
  );
}
