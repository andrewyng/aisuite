import { useState } from "react";

export interface Option {
  value: string;
  label: string;
  description?: string;
}

interface Props {
  prefix?: string;
  value: string;
  options: Option[];
  onChange: (value: string) => void;
  align?: "left" | "right";
}

export function Dropdown({ prefix, value, options, onChange, align = "left" }: Props) {
  const [open, setOpen] = useState(false);
  const current = options.find((o) => o.value === value);
  return (
    <div className="dd">
      <button className="pill" onClick={() => setOpen((v) => !v)}>
        {prefix ? `${prefix}: ` : ""}
        {current?.label || value} <span className="caret">▾</span>
      </button>
      {open && (
        <>
          <div className="dd-backdrop" onClick={() => setOpen(false)} />
          <div className={"dd-menu " + align}>
            {options.map((o) => (
              <div
                key={o.value}
                className={"dd-item" + (o.value === value ? " sel" : "")}
                onClick={() => {
                  onChange(o.value);
                  setOpen(false);
                }}
              >
                <div className="dd-label">
                  {o.label}
                  {o.value === value && <span className="chk">✓</span>}
                </div>
                {o.description && <div className="dd-desc">{o.description}</div>}
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
