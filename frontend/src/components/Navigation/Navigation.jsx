import { IoMdClose } from "react-icons/io";
import { IoMenu } from "react-icons/io5";
import "./Navigation.css";
import { useState } from "react";

const Navigation = () => {
  const [openMenu, setOpenMenu] = useState(false);

  const handleOpenMenu = () => {
    setOpenMenu(!openMenu);
  };

  return (
    <div className="navigation_container">
      {!openMenu && (
        <button className="open_menu_btn" onClick={handleOpenMenu}>
          <IoMenu />
        </button>
      )}

      <div
        className={
          openMenu
            ? "navigation_wrapper active container"
            : "navigation_wrapper container"
        }
      >
        <button className="close_btn" onClick={handleOpenMenu}>
          <IoMdClose />
        </button>

        <ul className="navigation_left">
          <li>Home</li>
          <li>Model Training</li>
          <li>Model Repository</li>
          <li>Privacy Risk Toolkit</li>
        </ul>
        <div className="navigation_right">
          <button className="sign_in">Sign in</button>
          <button className="sign_up">Sign up</button>
        </div>
      </div>
    </div>
  );
};

export default Navigation;
