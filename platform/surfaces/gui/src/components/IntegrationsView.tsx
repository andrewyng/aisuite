import { ConnectorsTab, McpTab } from "./ManageModal";
import { Icon } from "./Icon";

// Combined "Integrations" surface: messaging connectors + MCP servers in one place.
export function IntegrationsView() {
  return (
    <div className="main">
      <div className="sa-view-head">
        <div className="sa-view-title">
          <Icon name="plug" size={19} className="mark" /> Integrations
        </div>
        <div className="sa-view-sub">
          External accounts and tools OpenCoworker can use — messaging connectors and MCP servers, with
          per-tool controls and approval boundaries.
        </div>
      </div>
      <div className="main-scroll">
        <div className="page-panel">
          <div className="sa-sub">Connectors</div>
          <ConnectorsTab />
          <div className="sa-sub" style={{ marginTop: 26 }}>
            MCP servers
          </div>
          <McpTab />
        </div>
      </div>
    </div>
  );
}
